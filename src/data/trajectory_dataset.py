"""
VLN Trajectory Dataset — multi-step trajectory prediction.

Extends VLNSlidingWindowDataset with 24-step trajectory prediction,
3D actions (dx, dy, delta_yaw), trajectory augmentation, and progress.
"""

import json
import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .heatmap_geometry import compute_history_heatmap
from .sliding_window_dataset import VLNSlidingWindowDataset, _evict_from_page_cache
from .trajectory_utils import (
    apply_trajectory_augmentation,
    compute_history_rel_poses,
    get_trajectory_relative_to_frame,
    interpolate_and_resample_trajectory,
)

logger = logging.getLogger(__name__)


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
        image_size: tuple[int, int] = (224, 224),
        hm_size: tuple[int, int] = (64, 64),
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
        load_traj_images: bool = False,
        traj_image_size: tuple[int, int] = (224, 224),
        # FGR2R 子指令配置
        fgr2r_subinstr_path: str | None = None,
        use_subinstruction: bool = False,
        # Stage 2: 前视图+lookdown (InternNav aligned) vs 全景图 VLM 输入
        panoramic_vlm_input: bool = True,
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
        self.load_traj_images = load_traj_images
        self.traj_image_size = traj_image_size
        self.panoramic_vlm_input = panoramic_vlm_input

        # 加载 FGR2R 子指令映射表
        self.use_subinstruction = use_subinstruction
        self._fgr2r_mapping = {}
        if use_subinstruction:
            self._load_fgr2r_mapping(fgr2r_subinstr_path)

        logger.info(
            f"VLNTrajectoryDataset initialized: predict_horizon={predict_horizon}, "
            f"action_scale={action_scale}, trajectory_aug={self.enable_trajectory_augmentation}, "
            f"random_subseq={self.random_subsequence}, use_subinstr={self.use_subinstruction}, "
            f"load_traj_images={self.load_traj_images}, "
            f"panoramic_vlm_input={self.panoramic_vlm_input}"
        )

    def _load_traj_image_raw(self, clip_dir: Path, frame_idx: int,
                             direction: str = "front") -> np.ndarray:
        """Load a single frame as (H, W, 3) uint8 for DualVLN visual memory.

        Uses ``traj_image_size`` (default 224x224) and no colour augmentation
        so that the DINOv2 backbone gets clean inputs.

        Args:
            direction: which view to load.  Use ``"front_down"`` for the
                lookdown observation when available.
        """
        clip_idx = self._get_clip_idx(clip_dir)
        storage_format = self._get_storage_format(clip_idx)

        if storage_format == "chunks":
            try:
                raw = self._get_chunk_frame_array(clip_idx, frame_idx, "rgb", direction=direction)
            except KeyError:
                if direction != "front":
                    raw = self._get_chunk_frame_array(clip_idx, frame_idx, "rgb", direction="front")
                else:
                    raise
            image = self._decode_chunk_rgb(raw, clip_dir, frame_idx)
        else:
            dir_name = direction
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

        tw, th = self.traj_image_size
        if image.shape[:2] != (th, tw):
            image = cv2.resize(image, (tw, th))
        return image

    @staticmethod
    def _compute_pixel_goal(
        current_pose: np.ndarray,
        goal_pose: np.ndarray,
        img_size: int | tuple[int, int] = 256,
        depth_map: np.ndarray | None = None,
        depth_tolerance: float = 0.5,
    ) -> list[int] | None:
        """Project the goal position onto the current front-view image.

        Uses pinhole projection with HFOV=90° (Habitat convention:
        X right, Y up, -Z forward).

        Aligned with InternNav paper: when ``depth_map`` is provided,
        the projected point is checked against the depth buffer.  If the
        depth at the projected pixel is significantly closer than the
        goal distance, the goal is considered occluded and ``None`` is
        returned.

        Args:
            img_size: scalar (square) or (width, height) tuple.
            depth_map: (H, W) depth in metres.  Values <= 0 are treated
                as invalid / infinite depth (no occlusion).
            depth_tolerance: margin in metres — the goal is accepted if
                ``depth_at_pixel >= goal_distance - tolerance``.

        Returns:
            [u, v] integer pixel coordinates, or ``None`` if the goal
            is behind the camera, outside the image, or occluded.
        """
        if isinstance(img_size, (tuple, list)):
            img_w, img_h = int(img_size[0]), int(img_size[1])
        else:
            img_w = img_h = int(img_size)

        T_inv = np.linalg.inv(np.asarray(current_pose, dtype=np.float64))
        goal_world = np.array([
            goal_pose[0, 3], goal_pose[1, 3], goal_pose[2, 3], 1.0,
        ], dtype=np.float64)
        p_cam = T_inv @ goal_world

        x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        if z >= -0.1:
            return None

        z_depth = -z
        fx = img_w / 2.0
        fy = img_h / 2.0
        cx = img_w / 2.0
        cy = img_h / 2.0

        u_f = fx * x / z_depth + cx
        v_f = fy * (-y) / z_depth + cy

        if u_f < 0 or u_f >= img_w or v_f < 0 or v_f >= img_h:
            return None

        u = max(0, min(img_w - 1, round(u_f)))
        v = max(0, min(img_h - 1, round(v_f)))

        if depth_map is not None:
            dm = depth_map
            if dm.ndim == 3 and dm.shape[-1] == 1:
                dm = dm[:, :, 0]
            dh, dw = dm.shape[:2]
            du = round(u_f * dw / img_size)
            dv = round(v_f * dh / img_size)
            du = max(0, min(dw - 1, du))
            dv = max(0, min(dh - 1, dv))
            pixel_depth = float(dm[dv, du])
            if pixel_depth > 0 and pixel_depth < z_depth - depth_tolerance:
                return None

        return [u, v]

    def _load_fgr2r_mapping(self, fgr2r_path: str | None = None):
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
                with open(fgr2r_path) as f:
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
        poses: list[np.ndarray],
        current_t: int,
        subseq_end: int,
        subseq_start: int = 0,
    ) -> tuple[np.ndarray, float, float]:
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

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str, float]]:
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
            #    panoramic_vlm_input=True:  全景 4 视图 → VLM (Stage 1)
            #    panoramic_vlm_input=False: 前视图 + lookdown → VLM (Stage 2, InternNav)
            if self._is_panoramic and self.panoramic_vlm_input:
                current_frame = self._load_frame(clip_dir, current_t, direction="front")
                current_views = self._load_all_views(clip_dir, current_t)
                history_panoramas = self._load_history_panoramas(clip_dir, history_indices)
            else:
                current_frame = self._load_frame(clip_dir, current_t, direction="front")
                current_views = None
                history_panoramas = None

            # 5. 加载位姿
            poses = self._load_poses(clip_idx)
            history_poses = [poses[i] for i in history_indices]
            current_pose = poses[current_t]

            # 6. 加载当前帧深度（用于遮挡检测）
            current_depth = self._load_depth(clip_dir, current_t)

            # 7. 计算热力图
            img_size, K = self._load_intrinsics(clip_idx, clip_dir)

            hm_w, hm_h = self.hm_size

            gt_visibility = None
            if self._is_panoramic and self.panoramic_vlm_input:
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

            result = {
                "history_frames": history_frames,        # [K, 3, H, W]
                "current_frame": current_frame,          # [3, H, W] (front view)
                "heatmap": heatmap_tensor,               # [Hm, Wm] or [N, 4, Hm, Wm] (panoramic)
                "trajectory": trajectory_tensor,         # [predict_horizon, 3]
                "trajectory_valid": trajectory_valid,    # float
                "progress": progress,                    # float (0-1)
                "action": action_tensor,                 # [2]
                "action_valid": action_valid,            # float
                "discrete_action": discrete_action,      # int
                "is_stop": is_stop,                      # float
                "text": text,                            # str
            }

            # 12. traj_images for DualVLN visual memory [pixel_goal, current]
            if self.load_traj_images:
                goal_frame_idx = min(subseq_end - 1, T - 1)
                traj_view = "front_down"
                try:
                    goal_img = self._load_traj_image_raw(clip_dir, goal_frame_idx, direction=traj_view)
                    curr_img = self._load_traj_image_raw(clip_dir, current_t, direction=traj_view)
                except (FileNotFoundError, ValueError, KeyError, OSError):
                    th, tw = self.traj_image_size[1], self.traj_image_size[0]
                    goal_img = np.zeros((th, tw, 3), dtype=np.uint8)
                    curr_img = np.zeros((th, tw, 3), dtype=np.uint8)
                traj_imgs = np.stack([goal_img, curr_img], axis=0).astype(np.float32) / 255.0
                result["traj_images"] = torch.from_numpy(traj_imgs)  # [2, H, W, 3]

                # Pixel-goal: project the *farthest visible* future
                # waypoint onto the current front view, aligned with
                # InternNav's depth-based occlusion filtering.
                front_depth = self._load_depth(clip_dir, current_t, direction="front")
                _img_w = img_size[0] if isinstance(img_size, tuple) else img_size
                pg = None
                for fi in range(goal_frame_idx, current_t, -1):
                    pg = self._compute_pixel_goal(
                        current_pose, poses[fi],
                        img_size=_img_w,
                        depth_map=front_depth,
                    )
                    if pg is not None:
                        break
                if pg is not None:
                    result["pixel_goal"] = pg
                else:
                    result["trajectory_valid"] = 0.0

            if gt_visibility is not None:
                result["gt_visibility"] = gt_visibility  # [N, 4]
            if current_views is not None:
                result["current_views"] = current_views  # [4, 3, H, W]
            if history_panoramas is not None:
                result["history_panoramas"] = history_panoramas  # [N, 4, 3, H, W]

            # Stage 2 InternNav: front-view + lookdown for VLM
            if self._is_panoramic and not self.panoramic_vlm_input:
                try:
                    ld = self._load_frame(clip_dir, current_t, direction="front_down")
                except Exception:
                    ld = current_frame
                result["lookdown_frame"] = ld  # [3, H, W]

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
            return self._get_dummy_sample_trajectory()

    def _get_dummy_sample_trajectory(self) -> dict[str, Union[torch.Tensor, str, float, int]]:
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
    image_size: tuple[int, int] = (224, 224),
    hm_size: tuple[int, int] = (64, 64),
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

