"""
VLN Heatmap Dataset Adapter (vln_heatmap_adapter.py)
===================================================

DataLoader for reading standard training format and returning train.md expected fields.

This adapter reads the packed dataset format and provides the exact interface
expected by the training pipeline as specified in train.md.

NEW: Dual-head support - loads both history and future heatmaps with uniform sampling.

Returns:
    {"frames": Tensor[T_sampled,3,H,W],
     "text": str,
     "gt_heatmap_history": Tensor[T_sampled,Hm,Wm],
     "gt_heatmap_future": Tensor[T_sampled,Hm,Wm],
     "gt_validity_history": Tensor[T_sampled],
     "gt_validity_future": Tensor[T_sampled],
     "meta": Dict}
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class VLNHeatmapDataset(Dataset):
    """
    Dataset adapter for VLN heatmap training.

    Reads standard training format (from pack_dataset.py) and returns
    train.md compatible batch format.
    """

    def __init__(self,
                 root: str,
                 split: str,
                 frames_per_clip: int,
                 heatmap_per_clip: int,
                 image_size: Tuple[int, int] = (384, 384),
                 hm_size: Tuple[int, int] = (64, 64),
                 num_sample_frames: Optional[int] = None,
                 load_actions: bool = False):
        """
        Initialize VLN Heatmap Dataset.

        Args:
            root: Root directory of standard training format
            split: Dataset split ('train', 'val', 'test')
            frames_per_clip: Number of frames to load per clip (T) - deprecated, use num_sample_frames
            heatmap_per_clip: Expected number of heatmaps per clip (K) - deprecated
            image_size: Target image size (W, H)
            hm_size: Target heatmap size (W, H)
            num_sample_frames: Number of frames to uniformly sample from each clip (None = use all)
            load_actions: Whether to load action data (actions.npy) if available
        """
        self.root = Path(root)
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.heatmap_per_clip = heatmap_per_clip
        self.image_size = image_size  # (W, H)
        self.hm_size = hm_size        # (W, H)
        self.num_sample_frames = num_sample_frames  # NEW: uniform sampling
        self.load_actions = load_actions  # NEW: action loading

        # Enumerate all clips
        self.clips = self._enumerate_clips()

        logger.info(f"VLNHeatmapDataset initialized: {len(self.clips)} clips, "
                   f"num_sample_frames={num_sample_frames}, "
                   f"image_size={image_size}, hm_size={hm_size}, "
                   f"load_actions={load_actions}")

    def _enumerate_clips(self) -> List[Path]:
        """
        Enumerate all clip directories in the split.

        Directly scans the split directory (e.g., train/, val/) without relying on split.txt files.
        """
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        logger.info(f"Loading clips from split directory: {split_dir}")
        clips = []

        # Find all scene directories
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

        if len(scene_dirs) == 0:
            logger.warning(f"No scene directories found in {split_dir}")
            return clips

        for scene_dir in scene_dirs:
            # Find all clip directories in scene
            clip_dirs = sorted([d for d in scene_dir.iterdir()
                              if d.is_dir() and d.name.startswith('clip_')])
            clips.extend(clip_dirs)

        if len(clips) == 0:
            raise FileNotFoundError(f"No clips found in {split_dir}")

        logger.info(f"Found {len(clips)} clips in {len(scene_dirs)} scenes")
        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Load and return one training sample.

        Returns:
            dict: {
                "frames": Tensor[T_sampled, 3, H, W] - RGB frames (uniformly sampled)
                "text": str - Navigation instruction from R2R dataset
                "gt_heatmap_history": Tensor[T_sampled, Hm, Wm] - History heatmaps
                "gt_heatmap_future": Tensor[T_sampled, Hm, Wm] - Future heatmaps
                "gt_validity_history": Tensor[T_sampled] - History validity mask
                "gt_validity_future": Tensor[T_sampled] - Future validity mask
                "meta": Dict - Metadata with keyframe indices and reference path
            }
        """
        clip_dir = self.clips[idx]

        try:
            # Load metadata first (needed for frame loading)
            meta = self._load_metadata(clip_dir)

            # Load frames with optional uniform sampling
            frames, sample_indices = self._load_frames(clip_dir, meta)

            # Load dual heatmaps and masks
            gt_hm_hist, gt_hm_fut, mask_hist, mask_fut = self._load_dual_heatmaps(clip_dir, sample_indices)

            # Extract text instruction
            text = meta.get("instruction", "")

            # ⭐ FIX: Don't return full meta dict - it contains variable-length lists that break collate
            # Only return necessary scalar fields for training
            result = {
                "frames": frames,                      # [T_sampled, 3, H, W]
                "text": text,                          # R2R navigation instruction
                "gt_heatmap_history": gt_hm_hist,     # [T_sampled, Hm, Wm]
                "gt_heatmap_future": gt_hm_fut,       # [T_sampled, Hm, Wm]
                "gt_validity_history": mask_hist,      # [T_sampled]
                "gt_validity_future": mask_fut,        # [T_sampled]
                # "meta": meta                         # REMOVED: contains variable-length lists
            }
            
            # 🆕 Load actions if requested and available
            if self.load_actions:
                actions, actions_mask = self._load_actions(clip_dir, sample_indices)
                result["gt_actions"] = actions          # [T_sampled, 2] or None
                result["gt_actions_mask"] = actions_mask  # [T_sampled] or None
            
            return result

        except Exception as e:
            logger.error(f"Error loading clip {clip_dir}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return dummy data to avoid crash
            return self._get_dummy_sample()

    def _load_frames(self, clip_dir: Path, meta: Optional[Dict] = None) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load RGB frames from clip directory with optional uniform sampling.

        Args:
            clip_dir: Clip directory path
            meta: Metadata dictionary

        Returns:
            Tuple of (frames_tensor, sample_indices):
                - frames_tensor: [T_sampled, 3, H, W] sampled frames
                - sample_indices: [T_sampled] indices of sampled frames
        """
        rgb_dir = clip_dir / "rgb"

        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

        # Find all RGB files
        rgb_files = sorted(rgb_dir.glob("*.png"))

        if len(rgb_files) == 0:
            raise FileNotFoundError(f"No RGB files found in {rgb_dir}")

        total_frames = len(rgb_files)

        # Determine sampling strategy
        if self.num_sample_frames is not None:
            if total_frames >= self.num_sample_frames:
                # Uniform sampling
                sample_indices = np.linspace(0, total_frames - 1, self.num_sample_frames, dtype=int)
            else:
                # ⭐ FIX: If fewer frames than target, repeat last frame to maintain consistent batch size
                sample_indices = np.concatenate([
                    np.arange(total_frames),
                    np.full(self.num_sample_frames - total_frames, total_frames - 1)
                ]).astype(int)
            sampled_files = [rgb_files[i] for i in sample_indices]
        else:
            # Use all frames (no sampling) - but this can cause variable batch sizes!
            sample_indices = np.arange(total_frames, dtype=int)
            sampled_files = rgb_files

        # Load sampled frames
        frames = []
        target_w, target_h = self.image_size

        for rgb_file in sampled_files:
            # Load and resize image
            image = cv2.imread(str(rgb_file))
            if image is None:
                raise ValueError(f"Failed to load image: {rgb_file}")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to target size
            if image.shape[:2] != (target_h, target_w):
                image = cv2.resize(image, (target_w, target_h))

            # Convert to tensor and normalize to [0, 1]
            image_tensor = torch.from_numpy(image).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

            frames.append(image_tensor)

        # Stack frames
        frames_tensor = torch.stack(frames, dim=0)  # [T_sampled, C, H, W]

        return frames_tensor, sample_indices

    def _load_dual_heatmaps(self, clip_dir: Path, sample_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load dual heatmaps (history and future) with sampling.

        Args:
            clip_dir: Clip directory path
            sample_indices: Indices of sampled frames

        Returns:
            Tuple of (hm_history, hm_future, mask_history, mask_future):
                - hm_history: [T_sampled, Hm, Wm] history heatmaps
                - hm_future: [T_sampled, Hm, Wm] future heatmaps
                - mask_history: [T_sampled] history validity masks
                - mask_future: [T_sampled] future validity masks
        """
        # File paths
        hm_hist_file = clip_dir / "heatmaps_history.npy"
        hm_fut_file = clip_dir / "heatmaps_future.npy"
        mask_hist_file = clip_dir / "mask_history.npy"
        mask_fut_file = clip_dir / "mask_future.npy"

        # Check file existence
        if not hm_hist_file.exists():
            raise FileNotFoundError(f"History heatmaps not found: {hm_hist_file}")
        if not hm_fut_file.exists():
            raise FileNotFoundError(f"Future heatmaps not found: {hm_fut_file}")
        if not mask_hist_file.exists():
            raise FileNotFoundError(f"History mask not found: {mask_hist_file}")
        if not mask_fut_file.exists():
            raise FileNotFoundError(f"Future mask not found: {mask_fut_file}")

        # Load arrays
        hm_hist = np.load(hm_hist_file).astype(np.float32)  # [N, 64, 64]
        hm_fut = np.load(hm_fut_file).astype(np.float32)    # [N, 64, 64]
        mask_hist = np.load(mask_hist_file).astype(np.float32)  # [N]
        mask_fut = np.load(mask_fut_file).astype(np.float32)    # [N]

        # Apply sampling (align with frame sampling)
        hm_hist_sampled = hm_hist[sample_indices]  # [T_sampled, 64, 64]
        hm_fut_sampled = hm_fut[sample_indices]
        mask_hist_sampled = mask_hist[sample_indices]  # [T_sampled]
        mask_fut_sampled = mask_fut[sample_indices]

        # Convert to tensors
        hm_hist_tensor = torch.from_numpy(hm_hist_sampled)
        hm_fut_tensor = torch.from_numpy(hm_fut_sampled)
        mask_hist_tensor = torch.from_numpy(mask_hist_sampled)
        mask_fut_tensor = torch.from_numpy(mask_fut_sampled)

        # Resize heatmaps if needed
        T_sampled, Hm_orig, Wm_orig = hm_hist_tensor.shape
        target_w, target_h = self.hm_size

        if (Hm_orig, Wm_orig) != (target_h, target_w):
            hm_hist_tensor = self._resize_heatmaps(hm_hist_tensor, mask_hist_tensor, (target_h, target_w))
            hm_fut_tensor = self._resize_heatmaps(hm_fut_tensor, mask_fut_tensor, (target_h, target_w))

        return hm_hist_tensor, hm_fut_tensor, mask_hist_tensor, mask_fut_tensor

    def _resize_heatmaps(self, heatmaps: torch.Tensor, masks: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Resize heatmaps to target size with proper normalization.

        Args:
            heatmaps: [T, H_orig, W_orig] heatmaps
            masks: [T] validity masks
            target_size: (H_target, W_target)

        Returns:
            torch.Tensor: [T, H_target, W_target] resized heatmaps
        """
        T = heatmaps.shape[0]
        target_h, target_w = target_size
        resized_heatmaps = []

        for t in range(T):
            hm = heatmaps[t:t+1].unsqueeze(0)  # [1, 1, H_orig, W_orig]

            if masks[t] > 0.5:  # Valid heatmap
                # Resize using bilinear interpolation
                hm_resized = F.interpolate(hm, size=(target_h, target_w),
                                         mode='bilinear', align_corners=False)
                hm_resized = hm_resized.squeeze(0).squeeze(0)  # [H_target, W_target]

                # Renormalize to ensure sum=1 (probability distribution)
                hm_sum = hm_resized.sum()
                if hm_sum > 0:
                    hm_resized = hm_resized / hm_sum
            else:
                # Invalid heatmap - keep as zeros
                hm_resized = torch.zeros(target_h, target_w)

            resized_heatmaps.append(hm_resized)

        return torch.stack(resized_heatmaps, dim=0)

    def _load_actions(self, clip_dir: Path, sample_indices: np.ndarray) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load 2D continuous actions from clip directory.

        Args:
            clip_dir: Clip directory path
            sample_indices: Indices of sampled frames

        Returns:
            Tuple of (actions, actions_mask):
                - actions: [T_sampled, 2] 2D actions (dx, dy) or None if not found
                - actions_mask: [T_sampled] validity mask (all 1s if actions exist) or None
        """
        actions_file = clip_dir / "actions.npy"
        
        if not actions_file.exists():
            # Actions not available for this clip
            logger.debug(f"Actions file not found: {actions_file}")
            return None, None
        
        try:
            # Load actions array [T, 2]
            actions = np.load(actions_file).astype(np.float32)
            
            # Apply sampling (align with frame sampling)
            actions_sampled = actions[sample_indices]  # [T_sampled, 2]
            
            # Convert to tensor
            actions_tensor = torch.from_numpy(actions_sampled)  # [T_sampled, 2]
            
            # Create validity mask (all valid if file exists)
            actions_mask = torch.ones(len(sample_indices), dtype=torch.float32)
            
            return actions_tensor, actions_mask
            
        except Exception as e:
            logger.warning(f"Error loading actions from {actions_file}: {e}")
            return None, None

    def _load_metadata(self, clip_dir: Path) -> Dict:
        """Load metadata from clip directory."""
        meta_file = clip_dir / "meta.json"

        if not meta_file.exists():
            logger.warning(f"Metadata file not found: {meta_file}")
            return {"clip_dir": str(clip_dir)}

        with open(meta_file, 'r') as f:
            meta = json.load(f)

        # Add clip directory for reference
        meta["clip_dir"] = str(clip_dir)

        return meta

    def _get_dummy_sample(self) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Generate dummy sample for error cases."""
        target_w, target_h = self.image_size
        hm_w, hm_h = self.hm_size

        # Use num_sample_frames if set, otherwise use frames_per_clip
        T = self.num_sample_frames if self.num_sample_frames is not None else self.frames_per_clip

        result = {
            "frames": torch.zeros(T, 3, target_h, target_w),
            "text": "",
            "gt_heatmap_history": torch.zeros(T, hm_h, hm_w),
            "gt_heatmap_future": torch.zeros(T, hm_h, hm_w),
            "gt_validity_history": torch.zeros(T),
            "gt_validity_future": torch.zeros(T),
            # "meta": {"error": "Failed to load clip"}  # REMOVED: for consistency with normal return
        }
        
        # 🆕 Add dummy actions if load_actions is enabled
        if self.load_actions:
            result["gt_actions"] = torch.zeros(T, 2)
            result["gt_actions_mask"] = torch.zeros(T)
        
        return result


def create_heatmap_dataloader(
    root: str,
    split: str,
    frames_per_clip: int,
    heatmap_per_clip: int,
    image_size: Tuple[int, int] = (384, 384),
    hm_size: Tuple[int, int] = (64, 64),
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create DataLoader for VLN heatmap training.

    Args:
        root: Root directory of standard training format
        split: Dataset split ('train', 'val', 'test')
        frames_per_clip: Number of frames per clip (T)
        heatmap_per_clip: Number of heatmaps per clip (K)
        image_size: Target image size (W, H)
        hm_size: Target heatmap size (W, H)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader: Configured data loader
    """
    dataset = VLNHeatmapDataset(
        root=root,
        split=split,
        frames_per_clip=frames_per_clip,
        heatmap_per_clip=heatmap_per_clip,
        image_size=image_size,
        hm_size=hm_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataloader


# Testing and validation
def test_vln_heatmap_dataset():
    """Test VLN heatmap dataset implementation."""
    logger.info("Testing VLN heatmap dataset...")

    # Test with packed demo data
    try:
        dataset = VLNHeatmapDataset(
            root="./data/habitat_vln",
            split="train",
            frames_per_clip=8,
            heatmap_per_clip=4,
            image_size=(384, 384),
            hm_size=(64, 64)
        )

        if len(dataset) == 0:
            logger.warning("No data found - this is expected if pack_dataset.py hasn't been run")
            return False

        # Test loading one sample
        sample = dataset[0]

        # Validate structure
        required_keys = {"frames", "text", "gt_heatmaps", "mask", "meta"}
        assert set(sample.keys()) == required_keys, f"Missing keys: {required_keys - set(sample.keys())}"

        # Validate shapes
        frames = sample["frames"]
        gt_heatmaps = sample["gt_heatmaps"]
        mask = sample["mask"]

        assert frames.shape == (8, 3, 384, 384), f"Frames shape mismatch: {frames.shape}"
        assert gt_heatmaps.shape == (4, 64, 64), f"Heatmaps shape mismatch: {gt_heatmaps.shape}"
        assert mask.shape == (4,), f"Mask shape mismatch: {mask.shape}"

        # Validate data types
        assert frames.dtype == torch.float32, f"Frames dtype mismatch: {frames.dtype}"
        assert gt_heatmaps.dtype == torch.float32, f"Heatmaps dtype mismatch: {gt_heatmaps.dtype}"
        assert mask.dtype == torch.float32, f"Mask dtype mismatch: {mask.dtype}"

        # Validate ranges
        assert 0 <= frames.min() and frames.max() <= 1, f"Frames not in [0, 1]: [{frames.min()}, {frames.max()}]"
        assert 0 <= mask.min() and mask.max() <= 1, f"Mask not in [0, 1]: [{mask.min()}, {mask.max()}]"

        # Validate heatmap normalization for valid masks
        for k in range(4):
            if mask[k] > 0.5:  # Valid heatmap
                hm_sum = gt_heatmaps[k].sum().item()
                assert abs(hm_sum - 1.0) < 1e-3, f"Heatmap {k} not normalized: sum={hm_sum}"

        logger.info("✅ VLN heatmap dataset test passed!")
        logger.info(f"   Dataset size: {len(dataset)}")
        logger.info(f"   Sample shapes: frames={frames.shape}, heatmaps={gt_heatmaps.shape}, mask={mask.shape}")
        logger.info(f"   Valid heatmaps: {(mask > 0.5).sum()}/{len(mask)}")

        return True

    except Exception as e:
        logger.error(f"❌ VLN heatmap dataset test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests when executed directly
    logging.basicConfig(level=logging.INFO)
    test_vln_heatmap_dataset()