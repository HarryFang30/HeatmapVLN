"""
VLN Sliding Window Dataset — core training dataset.

Splits a video sequence into training samples via a sliding window.
Each sample: history frames + current frame -> heatmap + action.
"""

import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .augmentation import ColorJitterAugmentation, GaussianNoiseAugmentation, InternNavStyleAugmentation
from .heatmap_geometry import compute_history_heatmap
from .trajectory_utils import compute_history_rel_poses

logger = logging.getLogger(__name__)

_FADV_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def _evict_from_page_cache(filepath):
    """Advise the kernel to drop cached pages for this file.

    Uses posix_fadvise(FADV_DONTNEED) which works without root/SYS_ADMIN.
    This prevents page cache from accumulating in Docker cgroup-limited
    containers, which would otherwise count towards the memory limit and
    trigger the OOM killer.
    """
    try:
        fd = os.open(str(filepath), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, _FADV_DONTNEED)
        finally:
            os.close(fd)
    except Exception:
        logger.debug("posix_fadvise skipped for %s", filepath, exc_info=True)


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
        image_size: tuple[int, int] = (224, 224),
        hm_size: tuple[int, int] = (64, 64),
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
        chunk_direction: str = "front",  # chunks 模式下读取的视角
        chunk_cache_size: int = 6,  # chunks 模式下缓存的数组个数（worker 内）
        metadata_cache_size: int = 500,  # 元数据 LRU 缓存大小（clip 数量），防止 worker 内存无限增长
        defer_heatmap_to_gpu: bool = False,  # 兼容 train.py 传入（由 GPUHeatmapComputer 处理）
        load_history_frames: bool = True,
    ):
        self.root = Path(root).expanduser()
        self.defer_heatmap_to_gpu = defer_heatmap_to_gpu
        self.load_history_frames = load_history_frames
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
        self.chunk_direction = chunk_direction
        self.chunk_cache_size = max(1, int(chunk_cache_size))
        self.metadata_cache_size = max(50, int(metadata_cache_size))

        # 数据增强 (仅训练集启用)
        self.enable_augmentation = enable_augmentation and (split == 'train')
        if self.enable_augmentation:
            self.color_jitter = ColorJitterAugmentation(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5
            )
            self.gaussian_noise = GaussianNoiseAugmentation(std=8.0, p=0.3)
            self.internnav_aug = InternNavStyleAugmentation(p=0.5)
            logger.info(
                "Data augmentation enabled: ColorJitter + GaussianNoise "
                "+ InternNavStyle (posterize/sharpness/autocontrast)"
            )

        # 枚举所有 clips
        self.clips = self._enumerate_clips()

        # 预计算每个 clip 的有效帧范围（用于 clip-level 采样）
        self._clip_valid_frames = {}  # clip_idx -> list of valid frame indices
        self._precompute_valid_frames()

        # 预计算样本索引
        self.sample_index = []  # [(clip_idx, current_frame_idx), ...]
        self._poses_cache: OrderedDict = OrderedDict()
        self._meta_cache: OrderedDict = OrderedDict()

        self._build_sample_index()

        # chunks 模式相关缓存（每个 dataloader worker 独立）
        # 全部使用 OrderedDict + LRU 淘汰，防止在 worker 进程中内存无限增长
        self._clip_dir_to_idx = {str(clip_dir): i for i, clip_dir in enumerate(self.clips)}
        self._storage_format_cache: OrderedDict = OrderedDict()
        self._chunk_frame_lookup_cache: OrderedDict = OrderedDict()
        self._chunk_key_map_cache: OrderedDict = OrderedDict()
        self._chunk_array_cache: OrderedDict = OrderedDict()
        self._intrinsics_cache: OrderedDict = OrderedDict()

        self._depth_is_meters = self._detect_depth_format()
        self._is_panoramic = self._detect_panoramic()

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

    def _lru_put(self, cache: OrderedDict, key, value, max_size: int):
        """向 LRU 缓存写入条目，超过上限时淘汰最旧条目。"""
        if key in cache:
            cache.move_to_end(key)
            cache[key] = value
        else:
            cache[key] = value
            while len(cache) > max_size:
                cache.popitem(last=False)

    def _lru_get(self, cache: OrderedDict, key):
        """从 LRU 缓存读取条目，命中时更新访问顺序。返回 (value, hit)。"""
        if key in cache:
            cache.move_to_end(key)
            return cache[key], True
        return None, False

    def _detect_depth_format(self) -> bool:
        """自动检测深度图格式：米制 (True) 或归一化 [0,1] (False)"""
        if not self.load_depth or len(self.clips) == 0:
            return False
        try:
            clip_dir = self.clips[0]
            depth_npy = clip_dir / "depth" / "000000.npy"
            if depth_npy.exists():
                d = np.load(str(depth_npy))
                is_meters = float(d.max()) > 2.0
            else:
                chunk_dir = clip_dir / "chunks"
                if chunk_dir.exists():
                    chunk_file = sorted(chunk_dir.glob("chunk_*.npz"))[0]
                    with np.load(str(chunk_file), allow_pickle=False) as data:
                        for k in data.files:
                            if k.startswith("depth"):
                                d = data[k][0]
                                is_meters = float(d.max()) > 2.0
                                break
                        else:
                            return False
                else:
                    return False
            logger.info(f"Depth format: {'meters (no normalization)' if is_meters else 'normalized [0,1]'}")
            return is_meters
        except Exception as e:
            logger.warning(f"Could not detect depth format: {e}, assuming normalized")
            return False

    def _load_intrinsics(self, clip_idx: int, clip_dir: Path) -> tuple[tuple[int, int], np.ndarray | None]:
        val, hit = self._lru_get(self._intrinsics_cache, clip_idx)
        if hit:
            return val

        intrinsics_path = clip_dir / "intrinsics.json"
        K = None
        if intrinsics_path.exists():
            with open(intrinsics_path) as f:
                intrinsics = json.load(f)
            img_size = (intrinsics["width"], intrinsics["height"])
            if "K" in intrinsics:
                K = np.array(intrinsics["K"], dtype=np.float32)
        else:
            img_size = (640, 480)

        result = (img_size, K)
        self._lru_put(self._intrinsics_cache, clip_idx, result, self.metadata_cache_size)
        return result

    def _detect_panoramic(self) -> bool:
        """检测数据集是否包含全景多视角数据 (rgb_front, rgb_right, rgb_back, rgb_left)"""
        if len(self.clips) == 0:
            return False
        try:
            clip_dir = self.clips[0]
            chunk_dir = clip_dir / "chunks"
            if chunk_dir.exists():
                chunk_file = sorted(chunk_dir.glob("chunk_*.npz"))[0]
                with np.load(str(chunk_file), allow_pickle=False) as data:
                    keys = set(data.files)
                    required = {"rgb_front", "rgb_right", "rgb_back", "rgb_left"}
                    is_pano = required.issubset(keys)
            else:
                rgb_dir = clip_dir / "rgb"
                is_pano = all((rgb_dir / d).exists() for d in ["front", "right", "back", "left"])
            logger.info(f"Panoramic dataset: {is_pano}")
            return is_pano
        except Exception as e:
            logger.warning(f"Could not detect panoramic format: {e}, assuming single-view")
            return False


    @property
    def depth_is_meters(self) -> bool:
        """Whether depth data is in meters (True) or normalized [0,1] (False)"""
        return self._depth_is_meters

    def _enumerate_clips(self) -> list[Path]:
        """枚举所有 clip 目录

        支持两种目录结构：
        1. root/split/scene/clip_xxx（标准结构）
        2. root/scene/clip_xxx（无 split 层，自动按 scene 名 hash 划分 90% train / 10% val）
        """
        split_dir = self.root / self.split
        use_auto_split = False

        if split_dir.exists():
            search_dir = split_dir
        else:
            logger.info(f"Split directory {split_dir} not found, using root with auto-split")
            search_dir = self.root
            use_auto_split = True

        if not search_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {search_dir}")

        scene_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir()])

        if use_auto_split:
            import hashlib
            val_ratio = 0.1
            train_scenes, val_scenes = [], []
            for sd in scene_dirs:
                h = int(hashlib.md5(sd.name.encode()).hexdigest(), 16) % 100
                if h < int(val_ratio * 100):
                    val_scenes.append(sd)
                else:
                    train_scenes.append(sd)
            if not val_scenes and len(train_scenes) > 1:
                val_scenes.append(train_scenes.pop())

            if self.split == 'all':
                logger.info(f"Using all {len(scene_dirs)} scenes (split=all, no auto-split)")
            elif self.split in ('val', 'test'):
                scene_dirs = val_scenes
                logger.info(f"Auto-split: {len(train_scenes)} train scenes, {len(val_scenes)} val scenes (split={self.split})")
            else:
                scene_dirs = train_scenes
                logger.info(f"Auto-split: {len(train_scenes)} train scenes, {len(val_scenes)} val scenes (split={self.split})")

        clips = []
        for scene_dir in scene_dirs:
            clip_dirs = sorted([
                d for d in scene_dir.iterdir()
                if d.is_dir() and d.name.startswith('clip_')
            ])
            clips.extend(clip_dirs)

        if len(clips) == 0:
            raise FileNotFoundError(f"No clips found in {search_dir} (split={self.split})")

        logger.info(f"Found {len(clips)} clips in {len(scene_dirs)} scenes")
        return clips

    def _precompute_valid_frames(self):
        """预计算每个 clip 的有效帧索引列表（用于 clip-level 采样）

        有效帧 = 有足够历史帧的帧 (frame_idx >= min_history)
        同时验证 chunk 完整性，排除有损坏文件的 clip。
        """
        self._clip_valid_frames = {}
        skipped_corrupted = 0

        for clip_idx, clip_dir in enumerate(self.clips):
            try:
                meta_file = clip_dir / "meta.json"
                if not meta_file.exists():
                    continue

                with open(meta_file) as f:
                    meta = json.load(f)

                T = meta["num_frames"]

                chunks_dir = clip_dir / "chunks"
                if chunks_dir.exists():
                    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
                    has_corrupted = False
                    for cf in chunk_files:
                        try:
                            with np.load(cf, allow_pickle=True) as _:
                                pass
                        except Exception:
                            logger.warning(f"Corrupted chunk, excluding clip: {cf}")
                            has_corrupted = True
                            break
                    if has_corrupted:
                        skipped_corrupted += 1
                        continue

                valid_frames = list(range(self.min_history, T))

                if len(valid_frames) > 0:
                    self._clip_valid_frames[clip_idx] = valid_frames

            except Exception as e:
                logger.warning(f"Failed to precompute valid frames for clip {clip_dir}: {e}")
                continue

        if skipped_corrupted > 0:
            logger.warning(f"Excluded {skipped_corrupted} clips with corrupted chunk files")
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

    def _load_meta(self, clip_idx: int) -> dict:
        """加载并缓存 clip 元数据（LRU 淘汰）"""
        val, hit = self._lru_get(self._meta_cache, clip_idx)
        if hit:
            return val

        clip_dir = self.clips[clip_idx]
        meta_file = clip_dir / "meta.json"

        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")

        with open(meta_file) as f:
            meta = json.load(f)

        self._lru_put(self._meta_cache, clip_idx, meta, self.metadata_cache_size)
        return meta

    def _load_poses(self, clip_idx: int) -> list[np.ndarray]:
        """加载并缓存位姿数据（LRU 淘汰）"""
        if self.cache_poses:
            val, hit = self._lru_get(self._poses_cache, clip_idx)
            if hit:
                return val

        clip_dir = self.clips[clip_idx]
        storage_format = self._get_storage_format(clip_idx)

        if storage_format == "chunks":
            self._ensure_chunk_index(clip_idx)
            meta = self._load_meta(clip_idx)
            num_frames = int(meta["num_frames"])
            poses = []
            for frame_idx in range(num_frames):
                pose = self._get_chunk_frame_array(clip_idx, frame_idx, "pose")
                poses.append(np.array(pose, dtype=np.float32))
        else:
            poses_file = clip_dir / "poses.json"
            if not poses_file.exists():
                raise FileNotFoundError(f"Poses file not found: {poses_file}")
            with open(poses_file) as f:
                poses_list = json.load(f)
            poses = [np.array(p, dtype=np.float32) for p in poses_list]

        if self.cache_poses:
            self._lru_put(self._poses_cache, clip_idx, poses, self.metadata_cache_size)

        return poses

    def _get_storage_format(self, clip_idx: int) -> str:
        """自动识别 clip 的存储格式（frames/chunks）。"""
        val, hit = self._lru_get(self._storage_format_cache, clip_idx)
        if hit:
            return val

        clip_dir = self.clips[clip_idx]
        meta = self._load_meta(clip_idx)
        storage_format = str(meta.get("storage_format", "")).lower()

        if storage_format not in {"frames", "chunks"}:
            storage_format = "chunks" if (clip_dir / "chunks").exists() else "frames"

        self._lru_put(self._storage_format_cache, clip_idx, storage_format, self.metadata_cache_size)
        return storage_format

    def _get_clip_idx(self, clip_dir: Path) -> int:
        clip_key = str(clip_dir)
        if clip_key not in self._clip_dir_to_idx:
            raise KeyError(f"Unknown clip_dir: {clip_dir}")
        return self._clip_dir_to_idx[clip_key]

    def _ensure_chunk_index(self, clip_idx: int):
        """建立 frame -> (chunk_path, local_idx) 索引，并推断键名（LRU 淘汰）。"""
        _, hit_lookup = self._lru_get(self._chunk_frame_lookup_cache, clip_idx)
        _, hit_keymap = self._lru_get(self._chunk_key_map_cache, clip_idx)
        if hit_lookup and hit_keymap:
            return

        clip_dir = self.clips[clip_idx]
        chunks_dir = clip_dir / "chunks"
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")

        chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))
        if len(chunk_files) == 0:
            raise FileNotFoundError(f"No chunk files found in {chunks_dir}")

        frame_lookup: dict[int, tuple[str, int]] = {}
        key_map: dict[str, str] = {}

        for chunk_path in chunk_files:
            chunk_path_str = str(chunk_path)
            try:
                with np.load(chunk_path, allow_pickle=True) as chunk_data:
                    if "frame_ids" not in chunk_data:
                        logger.warning(f"frame_ids missing in chunk, skipping: {chunk_path}")
                        continue
                    frame_ids = np.array(chunk_data["frame_ids"], dtype=np.int32)
                    for local_idx, frame_id in enumerate(frame_ids.tolist()):
                        frame_lookup[int(frame_id)] = (chunk_path_str, int(local_idx))

                    if not key_map:
                        files = set(chunk_data.files)
                        for base_key in ("rgb", "depth", "pose"):
                            if base_key in files:
                                key_map[base_key] = base_key
                                continue
                            preferred_key = f"{base_key}_{self.chunk_direction}"
                            if preferred_key in files:
                                key_map[base_key] = preferred_key
                                continue
                            candidates = sorted([k for k in files if k.startswith(f"{base_key}_")])
                            if len(candidates) > 0:
                                key_map[base_key] = candidates[0]
            except Exception as e:
                logger.warning(f"Corrupted chunk file, skipping: {chunk_path} ({e})")

        if "rgb" not in key_map or "pose" not in key_map:
            raise KeyError(f"Chunk keys missing (need rgb/pose): clip={clip_dir}, key_map={key_map}")

        self._lru_put(self._chunk_frame_lookup_cache, clip_idx, frame_lookup, self.metadata_cache_size)
        self._lru_put(self._chunk_key_map_cache, clip_idx, key_map, self.metadata_cache_size)

    def _load_chunk_array(self, clip_idx: int, chunk_path: str, array_key: str) -> np.ndarray:
        """加载并缓存 chunk 内单个数组（LRU 淘汰）。"""
        cache_key = (clip_idx, chunk_path, array_key)
        val, hit = self._lru_get(self._chunk_array_cache, cache_key)
        if hit:
            return val

        with np.load(chunk_path, allow_pickle=True) as chunk_data:
            if array_key not in chunk_data:
                raise KeyError(f"Key {array_key} not found in chunk: {chunk_path}")
            arr = chunk_data[array_key]
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
        _evict_from_page_cache(chunk_path)

        self._lru_put(self._chunk_array_cache, cache_key, arr, self.chunk_cache_size)
        return arr

    def _get_chunk_frame_array(self, clip_idx: int, frame_idx: int, base_key: str,
                               direction: str | None = None) -> np.ndarray:
        """读取 chunks 模式下指定帧的单个字段（rgb/depth/pose）。

        Args:
            direction: 如果指定，直接用 '{base_key}_{direction}' 作为键，
                      绕过 key_map（用于加载非默认方向的数据）。
        """
        self._ensure_chunk_index(clip_idx)

        frame_lookup = self._chunk_frame_lookup_cache[clip_idx]
        if frame_idx not in frame_lookup:
            raise KeyError(f"frame {frame_idx} not found in chunk index for clip {clip_idx}")

        chunk_path, local_idx = frame_lookup[frame_idx]

        if direction:
            actual_key = f"{base_key}_{direction}"
        else:
            key_map = self._chunk_key_map_cache[clip_idx]
            if base_key not in key_map:
                raise KeyError(f"{base_key} key not found in chunk key_map for clip {clip_idx}")
            actual_key = key_map[base_key]

        arr = self._load_chunk_array(clip_idx, chunk_path, actual_key)
        return arr[local_idx]

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

    def _decode_chunk_rgb(self, raw, clip_dir, frame_idx) -> np.ndarray:
        """将 chunk 中的 rgb 数据解码为 RGB numpy 数组 [H, W, 3]

        支持多种存储格式：
        - [H, W, C] uint8 数组（已解码）
        - [N,] uint8 数组（JPEG 字节流）
        - [N,] object 数组（JPEG 字节以 int 对象存储，pickle 导致）
        - bytes / bytearray（原始 JPEG）
        """
        if isinstance(raw, np.ndarray):
            if raw.ndim == 3 and raw.shape[2] >= 3:
                return cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
            if raw.ndim == 1 and raw.dtype == np.uint8:
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode JPEG at clip={clip_dir}, frame={frame_idx}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if raw.ndim == 1 and raw.dtype == object:
                arr = np.array(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(raw, (bytes, bytearray)):
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise ValueError(
            f"Cannot decode chunk rgb at clip={clip_dir}, frame={frame_idx}, "
            f"type={type(raw).__name__}, shape={getattr(raw, 'shape', '?')}, dtype={getattr(raw, 'dtype', '?')}"
        )

    def _load_frame(self, clip_dir: Path, frame_idx: int,
                    apply_augmentation: bool = True,
                    direction: str | None = None) -> torch.Tensor:
        """加载单帧图像

        Args:
            direction: 指定方向 (front/right/back/left)，None 使用默认 chunk_direction
        """
        clip_idx = self._get_clip_idx(clip_dir)
        storage_format = self._get_storage_format(clip_idx)

        if storage_format == "chunks":
            raw = self._get_chunk_frame_array(clip_idx, frame_idx, "rgb", direction=direction)
            image = self._decode_chunk_rgb(raw, clip_dir, frame_idx)
        else:
            dir_name = direction or self.chunk_direction
            rgb_candidates = [
                clip_dir / "rgb" / f"{frame_idx:06d}.jpg",
                clip_dir / "rgb" / f"{frame_idx:06d}.png",
                clip_dir / "rgb" / dir_name / f"{frame_idx:06d}.jpg",
                clip_dir / "rgb" / dir_name / f"{frame_idx:06d}.png",
            ]
            rgb_path = next((p for p in rgb_candidates if p.exists()), None)
            if rgb_path is None:
                raise FileNotFoundError(f"RGB file not found: clip={clip_dir}, frame={frame_idx:06d}")

            image = cv2.imread(str(rgb_path))
            _evict_from_page_cache(rgb_path)
            if image is None:
                raise ValueError(f"Failed to load image: {rgb_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_w, target_h = self.image_size
        if image.shape[:2] != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h))

        if apply_augmentation and self.enable_augmentation:
            image = self.color_jitter(image)
            image = self.gaussian_noise(image)
            image = self.internnav_aug(image)

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

    PANORAMIC_DIRECTIONS = ["front", "right", "back", "left"]

    def _load_all_views(self, clip_dir: Path, frame_idx: int) -> torch.Tensor:
        """加载 4 个方向的独立图像。

        Returns:
            [4, C, H, W] tensor
        """
        views = []
        for d in self.PANORAMIC_DIRECTIONS:
            view = self._load_frame(clip_dir, frame_idx, apply_augmentation=True, direction=d)
            views.append(view)
        return torch.stack(views, dim=0)

    def _load_history_panoramas(
        self,
        clip_dir: Path,
        frame_indices: np.ndarray,
    ) -> torch.Tensor:
        """加载历史全景观测，返回 [N, 4, C, H, W]。"""
        panoramas = [
            self._load_all_views(clip_dir, int(frame_idx))
            for frame_idx in frame_indices
        ]
        return torch.stack(panoramas, dim=0)

    def _compute_per_history_multiview_heatmaps(
        self,
        clip_idx: int,
        clip_dir: Path,
        history_poses: list[np.ndarray],
        current_t: int,
        img_size: tuple[int, int],
        K: np.ndarray | None,
        hm_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按历史位置分别计算 4 视角 GT，返回 [N, 4, H, W] 和 [N, 4]。"""
        current_poses = {}
        current_depths = {}
        fallback_pose = self._load_poses(clip_idx)[current_t]
        hm_h, hm_w = hm_size

        for direction in self.PANORAMIC_DIRECTIONS:
            try:
                pose = self._get_chunk_frame_array(clip_idx, current_t, "pose", direction=direction)
                current_poses[direction] = np.array(pose, dtype=np.float32)
            except (KeyError, Exception):
                current_poses[direction] = fallback_pose
            current_depths[direction] = self._load_depth(clip_dir, current_t, direction=direction)

        per_history_heatmaps = []
        per_history_visibility = []
        for hist_pose in history_poses:
            view_heatmaps = []
            view_visibility = []
            for direction in self.PANORAMIC_DIRECTIONS:
                heatmap, visibility = compute_history_heatmap(
                    history_poses=[hist_pose],
                    current_pose=current_poses[direction],
                    current_depth=current_depths[direction],
                    hm_size=(hm_h, hm_w),
                    img_size=img_size,
                    K=K,
                    depth_normalize=not self._depth_is_meters,
                )
                view_heatmaps.append(torch.from_numpy(heatmap).float())
                view_visibility.append(float(visibility > 0))

            per_history_heatmaps.append(torch.stack(view_heatmaps, dim=0))
            per_history_visibility.append(torch.tensor(view_visibility, dtype=torch.float32))

        return torch.stack(per_history_heatmaps, dim=0), torch.stack(per_history_visibility, dim=0)

    def _load_depth(self, clip_dir: Path, frame_idx: int,
                    direction: str | None = None) -> np.ndarray | None:
        """加载深度图

        Args:
            direction: 指定方向 (front/right/back/left)，None 使用默认 chunk_direction
        """
        if not self.load_depth:
            return None

        clip_idx = self._get_clip_idx(clip_dir)
        storage_format = self._get_storage_format(clip_idx)

        if storage_format == "chunks":
            try:
                return np.array(self._get_chunk_frame_array(clip_idx, frame_idx, "depth",
                                                            direction=direction))
            except KeyError:
                return None

        dir_name = direction or self.chunk_direction
        depth_candidates = [
            clip_dir / "depth" / f"{frame_idx:06d}.npy",
            clip_dir / "depth" / dir_name / f"{frame_idx:06d}.npy",
        ]
        depth_path = next((p for p in depth_candidates if p.exists()), None)
        if depth_path is None:
            return None

        try:
            depth = np.load(depth_path)
            _evict_from_page_cache(depth_path)
            return depth
        except Exception as e:
            logger.warning(f"Failed to load depth: {depth_path}: {e}")
            return None

    def _load_actions(self, clip_dir: Path) -> np.ndarray | None:
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

    def _load_discrete_actions(self, clip_dir: Path) -> np.ndarray | None:
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

    def compute_action_stats(self, margin: float = 0.1) -> tuple[list[float], list[float]]:
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

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str, float]]:
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
            history_frames = (
                self._load_frames(clip_dir, history_indices)
                if self.load_history_frames else
                torch.zeros(1, 3, self.image_size[1], self.image_size[0])
            )

            # 4. 加载当前帧
            if self._is_panoramic:
                current_views = self._load_all_views(clip_dir, current_t)
                current_frame = current_views[0]
                history_panoramas = self._load_history_panoramas(clip_dir, history_indices)
            else:
                current_frame = self._load_frame(clip_dir, current_t)
                current_views = None
                history_panoramas = None

            # 5. 加载位姿
            poses = self._load_poses(clip_idx)
            history_poses = [poses[i] for i in history_indices]
            current_pose = poses[current_t]

            # 6. 加载当前帧深度（用于遮挡检测）
            current_depth = self._load_depth(clip_dir, current_t) if (self.load_depth and not self._is_panoramic and not self.defer_heatmap_to_gpu) else None

            # 7. 计算热力图
            img_size, K = self._load_intrinsics(clip_idx, clip_dir)

            hm_w, hm_h = self.hm_size

            gt_visibility = None
            if self._is_panoramic:
                heatmap_tensor, gt_visibility = self._compute_per_history_multiview_heatmaps(
                    clip_idx=clip_idx,
                    clip_dir=clip_dir,
                    history_poses=history_poses,
                    current_t=current_t,
                    img_size=img_size,
                    K=K,
                    hm_size=(hm_h, hm_w),
                )
            elif self.defer_heatmap_to_gpu:
                heatmap_tensor = torch.zeros(hm_h, hm_w)
            else:
                heatmap, _ = compute_history_heatmap(
                    history_poses=history_poses,
                    current_pose=current_pose,
                    current_depth=current_depth,
                    hm_size=(hm_h, hm_w),
                    img_size=img_size,
                    K=K,
                    depth_normalize=not self._depth_is_meters,
                )
                heatmap_tensor = torch.from_numpy(heatmap).float()

            # 8. 加载连续动作
            actions = self._load_actions(clip_dir)
            if actions is not None and current_t < len(actions):
                action = actions[current_t]
                if current_t == T - 1:
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
                discrete_action = int(discrete_actions[current_t])
                is_stop = 1.0 if discrete_action == 0 else 0.0
            else:
                discrete_action = 1
                is_stop = 0.0

            result = {
                "history_frames": history_frames,      # [K, 3, H, W]
                "current_frame": current_frame,        # [3, H, W] (front view)
                "heatmap": heatmap_tensor,             # [Hm, Wm] or [N, 4, Hm, Wm] (panoramic)
                "action": action_tensor,               # [2]
                "action_valid": action_valid,          # float
                "discrete_action": discrete_action,    # int (0-3)
                "is_stop": is_stop,                    # float (0 or 1)
                "text": text,                          # str
            }
            if gt_visibility is not None:
                result["gt_visibility"] = gt_visibility  # [N, 4]
            if current_views is not None:
                result["current_views"] = current_views  # [4, 3, H, W]
            if history_panoramas is not None:
                result["history_panoramas"] = history_panoramas  # [N, 4, 3, H, W]

            result["history_rel_poses"] = torch.from_numpy(
                compute_history_rel_poses(history_poses, current_pose)
            ).float()                                              # [K, 4]

            if self.defer_heatmap_to_gpu:
                result["history_poses"] = torch.from_numpy(
                    np.stack(history_poses, axis=0)).float()       # [K, 4, 4]
                result["current_pose"] = torch.from_numpy(
                    current_pose).float()                          # [4, 4]
                if current_depth is not None:
                    d = current_depth
                    if d.ndim == 3 and d.shape[-1] == 1:
                        d = d[:, :, 0]
                    result["current_depth"] = torch.from_numpy(
                        d.astype(np.float32))                      # [Hd, Wd]
                else:
                    result["current_depth"] = torch.zeros(1, 1)
                if K is not None:
                    result["intrinsics"] = torch.from_numpy(K)     # [3, 3]

            return result

        except Exception as e:
            logger.error(f"Error loading sample {idx} (clip {clip_idx}, t={current_t}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_dummy_sample()

    def _get_dummy_sample(self) -> dict[str, Union[torch.Tensor, str, float, int]]:
        """生成虚拟样本（用于错误处理）"""
        target_w, target_h = self.image_size
        hm_w, hm_h = self.hm_size
        K_heatmap = self.num_history_sample
        K_frames = self.num_history_sample if self.load_history_frames else 1

        if self._is_panoramic:
            result = {
                "history_frames": torch.zeros(K_frames, 3, target_h, target_w),
                "current_frame": torch.zeros(3, target_h, target_w),
                "current_views": torch.zeros(4, 3, target_h, target_w),
                "history_panoramas": torch.zeros(K_heatmap, 4, 3, target_h, target_w),
                "heatmap": torch.zeros(K_heatmap, 4, hm_h, hm_w),
                "gt_visibility": torch.zeros(K_heatmap, 4),
                "action": torch.zeros(2),
                "action_valid": 0.0,
                "discrete_action": 1,
                "is_stop": 0.0,
                "text": "",
            }
        else:
            result = {
                "history_frames": torch.zeros(K_frames, 3, target_h, target_w),
                "current_frame": torch.zeros(3, target_h, target_w),
                "heatmap": torch.zeros(hm_h, hm_w),
                "action": torch.zeros(2),
                "action_valid": 0.0,
                "discrete_action": 1,
                "is_stop": 0.0,
                "text": "",
            }
        result["history_rel_poses"] = torch.zeros(K_heatmap, 4)
        if self.defer_heatmap_to_gpu:
            result["history_poses"] = torch.zeros(K_heatmap, 4, 4)
            result["current_pose"] = torch.zeros(4, 4)
            result["current_depth"] = torch.zeros(1, 1)
        return result


def create_sliding_window_dataloader(
    root: str,
    split: str,
    min_history: int = 5,
    num_history_sample: int = 8,
    image_size: tuple[int, int] = (224, 224),
    hm_size: tuple[int, int] = (64, 64),
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
