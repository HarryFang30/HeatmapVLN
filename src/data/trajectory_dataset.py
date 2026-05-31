"""
VLN Trajectory Dataset — multi-step trajectory prediction.

Extends VLNSlidingWindowDataset with 24-step trajectory prediction,
3D actions (dx, dy, delta_yaw), trajectory augmentation, and progress.
"""

import json
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised in lightweight test envs
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from .heatmap_geometry import compute_history_heatmap
from .pano_view_pixel_goal import (
    PANO_HORIZONTAL_VIEWS,
    VIEW_STOP,
    VIEW_TURN,
    VIEW_TURN_LEFT,
    VIEW_TURN_RIGHT,
    load_intrinsics,
    resolve_farthest_pano_pixel_goal,
)
from .sliding_window_dataset import VLNSlidingWindowDataset, _evict_from_page_cache
from .trajectory_utils import (
    apply_trajectory_augmentation,
    compute_history_rel_poses,
    get_trajectory_relative_to_frame,
    interpolate_and_resample_trajectory,
)

logger = logging.getLogger(__name__)

from ._constants import SYSTEM2_ACTION_TEXT as _SYSTEM2_ACTION_TEXT


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv-python is required for trajectory image decoding") from _CV2_IMPORT_ERROR


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
        compute_pixel_goal: bool = False,
        load_lookdown_for_system2: bool = False,
        pixel_goal_direction: str = "front",
        load_history_heatmap: bool = True,
        require_sft_target: bool = False,
        sft_include_turns: bool = True,
        sft_include_forward: bool = False,
        sft_num_future_steps: int = 4,
        system2_sample_step: int = 4,
        system2_min_pixel_goal_len: int = 3,
        system2_stop_oversample: int = 5,
        include_stop_samples_random_subsequence: bool = False,
        # FGR2R 子指令配置
        fgr2r_subinstr_path: str | None = None,
        use_subinstruction: bool = False,
        # Stage 2: 前视图+lookdown (InternNav aligned) vs 全景图 VLM 输入
        panoramic_vlm_input: bool = True,
        compute_pano_view_pixel_goal: bool | None = None,
        pano_max_side_dist_m: float = 6.0,
    ):
        # ``VLNSlidingWindowDataset.__init__`` calls ``self._build_sample_index()``
        # before its chunk caches / panoramic detection fields are fully
        # initialized.  Since this class overrides that method for System2 SFT,
        # keep the parent bootstrap on the plain sliding-window path and rebuild
        # the InternNav index after ``super().__init__`` completes.
        actual_require_sft_target = require_sft_target
        self.require_sft_target = False
        self.include_stop_samples_random_subsequence = include_stop_samples_random_subsequence
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
            include_stop_samples_random_subsequence=include_stop_samples_random_subsequence,
        )

        self.predict_horizon = predict_horizon
        self.action_scale = action_scale
        self.enable_trajectory_augmentation = enable_trajectory_augmentation and (split == 'train')
        self.load_traj_images = load_traj_images
        self.compute_pixel_goal = compute_pixel_goal or load_traj_images
        self.load_lookdown_for_system2 = load_lookdown_for_system2
        self.pixel_goal_direction = pixel_goal_direction
        self.load_history_heatmap = load_history_heatmap
        self.require_sft_target = actual_require_sft_target
        from collections import OrderedDict
        self._directional_poses_cache: OrderedDict = OrderedDict()
        self.sft_include_turns = sft_include_turns
        self.sft_include_forward = sft_include_forward
        self.sft_num_future_steps = max(int(sft_num_future_steps), 1)
        self.system2_sample_step = max(int(system2_sample_step), 1)
        self.system2_min_pixel_goal_len = max(int(system2_min_pixel_goal_len), 1)
        self.system2_stop_oversample = max(int(system2_stop_oversample), 0)
        self.traj_image_size = traj_image_size
        self.traj_sequence_max_len = 12
        self.panoramic_vlm_input = panoramic_vlm_input
        if compute_pano_view_pixel_goal is None:
            compute_pano_view_pixel_goal = panoramic_vlm_input and self.compute_pixel_goal
        self.compute_pano_view_pixel_goal = bool(compute_pano_view_pixel_goal)
        self.pano_max_side_dist_m = float(pano_max_side_dist_m)

        # 加载 FGR2R 子指令映射表
        self.use_subinstruction = use_subinstruction
        self._fgr2r_mapping = {}
        if use_subinstruction:
            self._load_fgr2r_mapping(fgr2r_subinstr_path)

        if self.require_sft_target:
            self._build_sample_index()
            # Index building fills LRU caches by touching every clip's poses,
            # chunk arrays, and intrinsics.  Clear them now so fork()-ed
            # DataLoader workers don't each inherit a full copy — that would
            # multiply process memory by num_workers × n_gpus (up to ~64×).
            self._directional_poses_cache.clear()
            self._poses_cache.clear()
            self._chunk_array_cache.clear()
            import src.data.pano_view_pixel_goal as _pvpg
            _pvpg._intrinsics_cache.clear()

        logger.info(
            f"VLNTrajectoryDataset initialized: predict_horizon={predict_horizon}, "
            f"action_scale={action_scale}, trajectory_aug={self.enable_trajectory_augmentation}, "
            f"random_subseq={self.random_subsequence}, use_subinstr={self.use_subinstruction}, "
            f"load_traj_images={self.load_traj_images}, "
            f"compute_pixel_goal={self.compute_pixel_goal}, "
            f"load_lookdown_for_system2={self.load_lookdown_for_system2}, "
            f"pixel_goal_direction={self.pixel_goal_direction}, "
            f"load_history_heatmap={self.load_history_heatmap}, "
            f"require_sft_target={self.require_sft_target}, "
            f"sft_num_future_steps={self.sft_num_future_steps}, "
            f"system2_sample_step={self.system2_sample_step}, "
            f"system2_min_pixel_goal_len={self.system2_min_pixel_goal_len}, "
            f"system2_stop_oversample={self.system2_stop_oversample}, "
            f"panoramic_vlm_input={self.panoramic_vlm_input}, "
            f"compute_pano_view_pixel_goal={self.compute_pano_view_pixel_goal}, "
            f"pano_max_side_dist_m={self.pano_max_side_dist_m}"
        )

    def set_epoch(self, epoch: int):
        if self.require_sft_target:
            self._epoch = epoch
            self._rng = np.random.RandomState(42 + epoch)
            # InternNav SFT index is deterministic — same frames every epoch.
            # Only reshuffle instead of rebuilding all projection computations.
            if self.sample_index:
                indices = list(range(len(self.sample_index)))
                self._rng.shuffle(indices)
                self.sample_index = [self.sample_index[i] for i in indices]
                new_range = {
                    new_idx: self._sample_subsequence_range[old_idx]
                    for new_idx, old_idx in enumerate(indices)
                }
                self._sample_subsequence_range = new_range
                logger.info(
                    "[Epoch %d] Reshuffled %d InternNav-style System2 SFT samples "
                    "(skip rebuild)", epoch, len(self.sample_index),
                )
            return
        super().set_epoch(epoch)

    def _build_sample_index(self):
        if self.require_sft_target:
            self._build_internnav_sample_index()
            return
        super()._build_sample_index()

    @staticmethod
    def _align_internnav_discrete_actions(discrete_actions: np.ndarray) -> np.ndarray:
        """Match NavPixelGoalDataset: ``actions = item['actions'][1:] + [0]``."""
        if len(discrete_actions) <= 1:
            return discrete_actions
        return np.concatenate([discrete_actions[1:], np.array([0], dtype=discrete_actions.dtype)])

    def _system2_discrete_actions(self, discrete_actions: np.ndarray | None) -> np.ndarray | None:
        if discrete_actions is None:
            return None
        if self.require_sft_target:
            return self._align_internnav_discrete_actions(discrete_actions)
        return discrete_actions

    def _resolve_farthest_pixel_goal(
        self,
        clip_idx: int,
        clip_dir: Path,
        current_t: int,
        num_frames: int,
        img_size: int | tuple[int, int],
    ) -> tuple[int, list[int]] | None:
        """Match InternNav farthest visible pixel goal on ``goal.{pitch_2}`` poses.

        Returns ``(relative_goal_frame_id, [u, v])`` like parquet ``pixel_goals``,
        or ``None`` when no waypoint projects into the current lookdown view.
        """
        if not self.compute_pixel_goal or current_t >= num_frames - 1:
            return None

        pg_direction = self.pixel_goal_direction or "front"
        pg_poses = self._load_poses(clip_idx)
        if pg_direction != "front":
            pg_poses = self._load_poses_for_direction(clip_idx, pg_direction)
        pg_depth = self._load_depth(clip_dir, current_t, direction=pg_direction)

        if isinstance(img_size, (tuple, list)):
            proj_size: int | tuple[int, int] = (int(img_size[0]), int(img_size[1]))
        else:
            proj_size = int(img_size)

        # InternNav precomputes relative_goal_frame_id on the full episode, not a random subsequence.
        for fi in range(num_frames - 1, current_t, -1):
            pg = self._compute_pixel_goal(
                pg_poses[current_t],
                pg_poses[fi],
                img_size=proj_size,
                depth_map=pg_depth,
            )
            if pg is None:
                continue
            goal_len = fi - current_t
            if goal_len < self.system2_min_pixel_goal_len:
                return None
            return goal_len, pg
        return None

    def _resolve_farthest_pano_pixel_goal(
        self,
        clip_idx: int,
        clip_dir: Path,
        current_t: int,
        num_frames: int,
        img_size: int | tuple[int, int],
    ) -> tuple[int, str, list[int], list[int] | None] | None:
        """Online C3 pano label: ``(goal_len, view_id, [u,v], legacy_front_uv)``."""
        if not self.compute_pano_view_pixel_goal or current_t >= num_frames - 1:
            return None

        if isinstance(img_size, (tuple, list)):
            proj_size: int | tuple[int, int] = (int(img_size[0]), int(img_size[1]))
        else:
            proj_size = int(img_size)

        poses_by_view = {
            direction: self._load_poses_for_direction(clip_idx, direction)
            for direction in PANO_HORIZONTAL_VIEWS
        }
        depth_front = self._load_depth(clip_dir, current_t, direction="front")
        try:
            intrinsics = load_intrinsics(clip_dir)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            intrinsics = None

        pano_goal = resolve_farthest_pano_pixel_goal(
            current_t=current_t,
            num_frames=num_frames,
            poses_by_view=poses_by_view,
            depth_front=depth_front,
            img_size=proj_size,
            intrinsics=intrinsics,
            min_goal_len=self.system2_min_pixel_goal_len,
            max_side_dist_m=self.pano_max_side_dist_m,
        )
        if pano_goal is None:
            return None
        goal_len, canonical, legacy_uv = pano_goal
        return goal_len, canonical.view_id, [canonical.u, canonical.v], legacy_uv

    def _internnav_sft_frame_kind(
        self,
        clip_idx: int,
        clip_dir: Path,
        frame_id: int,
        num_frames: int,
        discrete_actions: np.ndarray,
    ) -> str | None:
        """Return ``pixel`` / ``turn`` / ``stop`` / ``None`` (skip), mirroring NavPixelGoalDataset."""
        if num_frames < 4:
            return None
        if frame_id == num_frames - 1:
            return "stop"

        action_flag = int(discrete_actions[frame_id])
        if self.compute_pano_view_pixel_goal:
            pg_result = self._resolve_farthest_pano_pixel_goal(
                clip_idx, clip_dir, frame_id, num_frames, self.image_size,
            )
        else:
            pg_result = self._resolve_farthest_pixel_goal(
                clip_idx, clip_dir, frame_id, num_frames, self.image_size,
            )
        if pg_result is None:
            if action_flag == 1:
                return None
            return "turn"
        return "pixel"

    def _build_internnav_sample_index(self):
        """Build sample index like ``NavPixelGoalDataset`` in internvla_n1_lerobot_dataset.py."""
        self.sample_index = []
        self._sample_subsequence_range = {}
        sample_step = self.system2_sample_step
        stop_repeat = self.system2_stop_oversample
        pixel_samples = 0
        turn_samples = 0
        stop_samples = 0
        skipped = 0

        logger.info(
            "Building InternNav SFT sample index across %d clips (sample_step=%d, min_goal_len=%d)...",
            len(self.clips), sample_step, self.system2_min_pixel_goal_len,
        )
        for clip_idx, clip_dir in enumerate(
            tqdm(self.clips, desc="Building SFT index", unit="clip", mininterval=5.0)
        ):
            try:
                meta = self._load_meta(clip_idx)
                num_frames = int(meta["num_frames"])
                if num_frames < 4:
                    skipped += 1
                    continue

                raw_actions = self._load_discrete_actions(clip_dir)
                if raw_actions is None or len(raw_actions) != num_frames:
                    skipped += 1
                    continue
                actions = self._align_internnav_discrete_actions(raw_actions)
                actions_len = len(actions)

                num_rounds = actions_len // sample_step
                for n in range(num_rounds + 1):
                    start_frame_id = n * sample_step
                    if (
                        start_frame_id == actions_len
                        or start_frame_id == actions_len - 1
                        or start_frame_id < self.min_history
                    ):
                        continue

                    kind = self._internnav_sft_frame_kind(
                        clip_idx, clip_dir, start_frame_id, num_frames, actions,
                    )
                    if kind is None:
                        continue
                    if kind == "turn" and not self.sft_include_turns:
                        continue
                    if self.load_traj_images and kind != "pixel":
                        continue
                    sample_idx = len(self.sample_index)
                    self.sample_index.append((clip_idx, start_frame_id))
                    self._sample_subsequence_range[sample_idx] = (0, num_frames)
                    if kind == "pixel":
                        pixel_samples += 1
                    elif kind == "turn":
                        turn_samples += 1

                last_frame = num_frames - 1
                if not self.load_traj_images and last_frame >= self.min_history:
                    for _ in range(stop_repeat):
                        sample_idx = len(self.sample_index)
                        self.sample_index.append((clip_idx, last_frame))
                        self._sample_subsequence_range[sample_idx] = (0, num_frames)
                        stop_samples += 1
            except Exception as exc:
                logger.warning("Failed to build InternNav SFT index for %s: %s", clip_dir, exc)
                skipped += 1

        if self.sample_index:
            indices = list(range(len(self.sample_index)))
            self._rng.shuffle(indices)
            self.sample_index = [self.sample_index[i] for i in indices]
            new_range = {
                new_idx: self._sample_subsequence_range[old_idx]
                for new_idx, old_idx in enumerate(indices)
            }
            self._sample_subsequence_range = new_range

        logger.info(
            "Built InternNav-style System2 SFT index: %s samples "
            "(pixel=%s, turn=%s, stop=%s, skipped_clips=%s, sample_step=%s, min_goal_len=%s)",
            len(self.sample_index),
            pixel_samples,
            turn_samples,
            stop_samples,
            skipped,
            sample_step,
            self.system2_min_pixel_goal_len,
        )

    def _result_has_system2_sft_target(self, result: dict[str, Union[torch.Tensor, str, float]]) -> bool:
        pano_kind = str(result.get("pano_sample_kind") or "").lower()
        pano_view_id = str(result.get("pano_view_id") or "")

        if pano_kind == "pixel" or result.get("pano_pixel_goal") is not None:
            if result.get("pano_pixel_goal") is None:
                return False
            goal_len = result.get("pano_pixel_goal_relative_len")
            if goal_len is None:
                return True
            return float(goal_len) >= self.system2_min_pixel_goal_len

        if pano_kind == "stop" or pano_view_id == VIEW_STOP:
            return True

        if pano_kind in ("turn", "turn_left", "turn_right") or pano_view_id in (VIEW_TURN, VIEW_TURN_LEFT, VIEW_TURN_RIGHT):
            return bool(self.sft_include_turns)

        if result.get("is_stop", 0.0) > 0.5 or int(result.get("discrete_action", 1)) == 0:
            return True

        pg = result.get("pixel_goal")
        if pg is not None:
            goal_len = result.get("pixel_goal_relative_len")
            return goal_len is None or goal_len >= self.system2_min_pixel_goal_len

        # NavPixelGoalDataset: skip forward-only frames when pixel_goal[0] == -1.
        if int(result.get("discrete_action", 1)) == 1:
            return bool(self.sft_include_forward)

        if self.sft_include_turns and result.get("turn_actions"):
            return True

        discrete_action = int(result.get("discrete_action", 1))
        if self.sft_include_turns and discrete_action in (2, 3, 5):
            return True
        return self.sft_include_forward and discrete_action == 1

    def _candidate_retry_indices(self, idx: int, max_attempts: int = 64) -> list[int]:
        total = len(self.sample_index)
        if total <= 1:
            return [idx]

        target_count = min(max(total, 1), max(max_attempts, 1))
        candidates = [idx]
        seen = {idx}
        seed = ((self._epoch + 1) * 1_000_003 + idx) % (2**32 - 1)
        rng = np.random.RandomState(seed)

        while len(candidates) < target_count and len(seen) < total:
            candidate = int(rng.randint(0, total))
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)

        return candidates

    def _collect_turn_actions(
        self,
        discrete_actions: np.ndarray | None,
        current_t: int,
    ) -> list[int]:
        """Match InternNav turn-label construction: collect turns until next forward."""
        if discrete_actions is None or current_t >= len(discrete_actions):
            return []

        current_action = int(discrete_actions[current_t])
        if current_action in (0, 1):
            return []

        end_t = min(len(discrete_actions), current_t + self.sft_num_future_steps)
        turn_actions: list[int] = []
        for t in range(current_t, end_t):
            action_code = int(discrete_actions[t])
            if action_code == 1:
                break
            if action_code not in _SYSTEM2_ACTION_TEXT:
                logger.debug(
                    "Skip unsupported System2 turn action %s at t=%s",
                    action_code,
                    t,
                )
                break
            turn_actions.append(action_code)
        return turn_actions

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
            _require_cv2()
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

    def _load_poses_for_direction(self, clip_idx: int, direction: str) -> list[np.ndarray]:
        """Load per-frame camera poses for a specific panoramic direction (LRU cached)."""
        cache_key = (clip_idx, direction)
        if self.cache_poses:
            val, hit = self._lru_get(self._directional_poses_cache, cache_key)
            if hit:
                return val

        try:
            meta = self._load_meta(clip_idx)
            num_frames = int(meta["num_frames"])
            poses = [
                np.array(
                    self._get_chunk_frame_array(clip_idx, frame_idx, "pose", direction=direction),
                    dtype=np.float32,
                )
                for frame_idx in range(num_frames)
            ]

            if self.cache_poses:
                self._lru_put(self._directional_poses_cache, cache_key, poses, self.metadata_cache_size)
            return poses
        except Exception:
            return self._load_poses(clip_idx)

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
            du = round(u_f * dw / img_w)
            dv = round(v_f * dh / img_h)
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
        camera_deg: float = 0,
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
            relative_xyyaw = get_trajectory_relative_to_frame(future_poses_np, camera_deg=camera_deg)

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

        except (ValueError, np.linalg.LinAlgError, IndexError) as e:
            logger.warning(f"Failed to compute trajectory: {e}", exc_info=True)
            resampled_poses = np.zeros((self.predict_horizon, 3), dtype=np.float32)
            trajectory_valid = 0.0

        return resampled_poses.astype(np.float32), trajectory_valid, progress

    def _build_sample(self, idx: int) -> dict[str, Union[torch.Tensor, str, float]]:
        """构建单个轨迹样本。异常向上传递，由调用方决定是否重试。"""
        clip_idx, current_t = self.sample_index[idx]
        clip_dir = self.clips[clip_idx]
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
        action_poses = poses
        action_camera_deg = 0.0
        if self.load_traj_images:
            action_poses = self._load_poses_for_direction(clip_idx, "front_down")
            action_camera_deg = float(meta.get("lookdown_pitch_deg", 30.0))
        history_poses = [poses[i] for i in history_indices]
        current_pose = poses[current_t]

        # 6. 仅在需要热力图监督时才加载正视角深度
        current_depth = None
        if self.load_history_heatmap and not self._is_panoramic:
            current_depth = self._load_depth(clip_dir, current_t)

        # 7. 计算热力图 / 生成占位张量
        img_size, K = self._load_intrinsics(clip_idx, clip_dir)
        hm_w, hm_h = self.hm_size

        gt_visibility = None
        if self.load_history_heatmap:
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
        elif self._is_panoramic and self.panoramic_vlm_input:
            heatmap_tensor = torch.zeros(len(history_indices), 4, hm_h, hm_w)
        else:
            heatmap_tensor = torch.zeros(hm_h, hm_w)

        # 8. 计算轨迹（使用子序列范围计算 progress）
        trajectory, trajectory_valid, progress = self._compute_trajectory(
            action_poses, current_t, subseq_end, subseq_start,
            camera_deg=action_camera_deg,
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

        # 11. 离散动作（InternNav SFT 使用 actions[1:]+[0] 对齐）
        raw_discrete_actions = self._load_discrete_actions(clip_dir)
        system2_actions = self._system2_discrete_actions(raw_discrete_actions)
        if system2_actions is not None and current_t < len(system2_actions):
            discrete_action = int(system2_actions[current_t])
            is_stop = 1.0 if discrete_action == 0 else 0.0
        else:
            discrete_action = 1
            is_stop = 0.0
        turn_actions = self._collect_turn_actions(system2_actions, current_t)

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
        if turn_actions:
            result["turn_actions"] = turn_actions
            result["turn_action_text"] = "".join(
                _SYSTEM2_ACTION_TEXT[action_code] for action_code in turn_actions
            )

        # Pixel-goal SFT / bridge target: farthest visible waypoint in lookdown view.
        # Mirrors InternNav parquet ``[relative_goal_frame_id, goal.{pitch_2}deg]``.
        pg_result = self._resolve_farthest_pixel_goal(
            clip_idx=clip_idx,
            clip_dir=clip_dir,
            current_t=current_t,
            num_frames=T,
            img_size=img_size,
        )
        if pg_result is not None:
            goal_len, pg = pg_result
            result["pixel_goal"] = pg
            result["pixel_goal_relative_len"] = goal_len
        elif self.load_traj_images and not self.compute_pano_view_pixel_goal:
            tv = result.get("trajectory_valid", 0.0)
            result["trajectory_valid"] = (
                torch.zeros_like(tv) if torch.is_tensor(tv) else 0.0
            )

        if self.compute_pano_view_pixel_goal:
            pano_result = self._resolve_farthest_pano_pixel_goal(
                clip_idx=clip_idx,
                clip_dir=clip_dir,
                current_t=current_t,
                num_frames=T,
                img_size=img_size,
            )
            if pano_result is not None:
                goal_len, view_id, pano_pg, legacy_uv = pano_result
                result["pano_view_id"] = view_id
                result["pano_pixel_goal"] = pano_pg
                result["pano_pixel_goal_relative_len"] = goal_len
                result["pano_sample_kind"] = "pixel"
                if legacy_uv is not None:
                    result["legacy_front_pixel_goal"] = legacy_uv
            elif float(result.get("is_stop", 0.0)) > 0.5:
                result["pano_view_id"] = VIEW_STOP
                result["pano_sample_kind"] = "stop"
            elif turn_actions or int(result.get("discrete_action", 1)) in (2, 3, 5):
                da = int(result.get("discrete_action", 1))
                if da == 2:
                    result["pano_view_id"] = VIEW_TURN_LEFT
                    result["pano_sample_kind"] = "turn_left"
                elif da == 3:
                    result["pano_view_id"] = VIEW_TURN_RIGHT
                    result["pano_sample_kind"] = "turn_right"
                else:
                    result["pano_view_id"] = VIEW_TURN
                    result["pano_sample_kind"] = "turn"

        goal_rel_len = result.get(
            "pixel_goal_relative_len",
            result.get("pano_pixel_goal_relative_len", 0),
        )
        goal_frame_idx = current_t + int(goal_rel_len or 0)
        has_system1_goal = (
            result.get("pixel_goal") is not None
            or (
                str(result.get("pano_sample_kind") or "").lower() == "pixel"
                and result.get("pano_pixel_goal") is not None
            )
        )
        if not has_system1_goal:
            goal_frame_idx = min(subseq_end - 1, T - 1)

        # 12. traj_images for DualVLN visual memory.
        # InternNav stores a sequence of lookdown frames from the System 2
        # trigger point onward; the action head repeats the first frame as
        # the fixed anchor and pairs it with each current frame.  The pixel
        # goal itself is carried by the generated text / latent queries,
        # not by replacing the anchor with a privileged future goal view.
        if self.load_traj_images:
            if not has_system1_goal:
                th, tw = self.traj_image_size[1], self.traj_image_size[0]
                traj_imgs_list = [np.zeros((th, tw, 3), dtype=np.uint8)]
                traj_poses_list = [np.zeros((self.predict_horizon, 3), dtype=np.float32)]
                traj_valid_list = [0.0]
            else:
                traj_view = "front_down"
                goal_len = max(goal_frame_idx - current_t, 1)
                interval = 2
                frame_offsets = np.arange(0, goal_len, interval, dtype=np.int32)
                if len(frame_offsets) == 0:
                    frame_offsets = np.array([0], dtype=np.int32)
                if len(frame_offsets) > self.traj_sequence_max_len:
                    interval = int(np.ceil(goal_len / self.traj_sequence_max_len))
                    frame_offsets = np.arange(0, goal_len, interval, dtype=np.int32)[:self.traj_sequence_max_len]

                traj_imgs_list = []
                traj_poses_list = []
                traj_valid_list = []
                try:
                    for offset in frame_offsets:
                        frame_idx = min(current_t + int(offset), goal_frame_idx)
                        curr_img = self._load_traj_image_raw(clip_dir, frame_idx, direction=traj_view)
                        traj_i, valid_i, _progress_i = self._compute_trajectory(
                            action_poses, frame_idx, subseq_end, current_t,
                            camera_deg=action_camera_deg,
                        )
                        if self.enable_trajectory_augmentation and valid_i > 0:
                            traj_i = apply_trajectory_augmentation(traj_i, p=0.5)
                        traj_imgs_list.append(curr_img)
                        traj_poses_list.append(traj_i)
                        traj_valid_list.append(valid_i)
                except (FileNotFoundError, ValueError, KeyError, OSError):
                    traj_imgs_list = []

                if not traj_imgs_list:
                    th, tw = self.traj_image_size[1], self.traj_image_size[0]
                    traj_imgs_list = [np.zeros((th, tw, 3), dtype=np.uint8)]
                    traj_poses_list = [np.zeros((self.predict_horizon, 3), dtype=np.float32)]
                    traj_valid_list = [0.0]

            pad_len = self.traj_sequence_max_len - len(traj_imgs_list)
            if pad_len > 0:
                traj_imgs_list.extend([traj_imgs_list[-1].copy() for _ in range(pad_len)])
                traj_poses_list.extend([traj_poses_list[-1].copy() for _ in range(pad_len)])
                traj_valid_list.extend([0.0 for _ in range(pad_len)])

            traj_imgs = np.stack(traj_imgs_list[:self.traj_sequence_max_len], axis=0).astype(np.float32) / 255.0
            traj_poses = np.stack(traj_poses_list[:self.traj_sequence_max_len], axis=0).astype(np.float32)
            traj_valids = np.asarray(traj_valid_list[:self.traj_sequence_max_len], dtype=np.float32)
            result["traj_images"] = torch.from_numpy(traj_imgs)  # [N, H, W, 3]
            result["trajectory"] = torch.from_numpy(traj_poses)  # [N, predict_horizon, 3]
            result["trajectory_valid"] = torch.from_numpy(traj_valids)  # [N]

        if gt_visibility is not None:
            result["gt_visibility"] = gt_visibility  # [N, 4]
        if current_views is not None:
            result["current_views"] = current_views  # [4, 3, H, W]
        if history_panoramas is not None:
            result["history_panoramas"] = history_panoramas  # [N, 4, 3, H, W]

        # InternNav-style System2 protocol needs a lookdown observation
        # after the first assistant emits ↓.  For panoramic VLM input this
        # keeps the first turn panoramic while matching InternNav's second
        # user turn.
        if self._is_panoramic and (
            not self.panoramic_vlm_input or self.load_lookdown_for_system2
        ):
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

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str, float]]:
        """
        加载一个训练样本（带轨迹）
        """
        if not self.require_sft_target:
            return self._build_sample(idx)

        errors: list[str] = []
        missing_target_indices: list[int] = []
        last_exception: Exception | None = None

        for candidate_idx in self._candidate_retry_indices(idx):
            try:
                result = self._build_sample(candidate_idx)
            except Exception as exc:
                last_exception = exc
                clip_idx, current_t = self.sample_index[candidate_idx]
                errors.append(
                    f"idx={candidate_idx} clip={clip_idx} t={current_t} err={exc!r}"
                )
                continue

            if self._result_has_system2_sft_target(result):
                return result
            missing_target_indices.append(candidate_idx)

        details = []
        if missing_target_indices:
            details.append(
                f"no_target={missing_target_indices[:8]}"
                + ("..." if len(missing_target_indices) > 8 else "")
            )
        if errors:
            details.append(
                f"errors={errors[:4]}"
                + ("..." if len(errors) > 4 else "")
            )
        detail_str = "; ".join(details) if details else "no candidates available"

        failure = RuntimeError(
            "Failed to produce a valid System2 SFT sample after bounded retries. "
            f"requested_idx={idx}; {detail_str}"
        )
        if last_exception is not None:
            raise failure from last_exception
        raise failure

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
