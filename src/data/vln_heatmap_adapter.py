"""
VLN Heatmap Dataset Adapter (vln_heatmap_adapter.py)
===================================================

DataLoader for reading standard training format and returning train.md expected fields.

This adapter reads the packed dataset format and provides the exact interface
expected by the training pipeline as specified in train.md.

Returns:
    {"frames": Tensor[T,3,H,W], "text": None, "gt_heatmaps": Tensor[K,Hm,Wm],
     "mask": Tensor[K], "meta": Dict}
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
                 hm_size: Tuple[int, int] = (64, 64)):
        """
        Initialize VLN Heatmap Dataset.

        Args:
            root: Root directory of standard training format
            split: Dataset split ('train', 'val', 'test')
            frames_per_clip: Number of frames to load per clip (T)
            heatmap_per_clip: Expected number of heatmaps per clip (K)
            image_size: Target image size (W, H)
            hm_size: Target heatmap size (W, H)
        """
        self.root = Path(root)
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.heatmap_per_clip = heatmap_per_clip
        self.image_size = image_size  # (W, H)
        self.hm_size = hm_size        # (W, H)

        # Enumerate all clips
        self.clips = self._enumerate_clips()

        logger.info(f"VLNHeatmapDataset initialized: {len(self.clips)} clips, "
                   f"frames={frames_per_clip}, heatmaps={heatmap_per_clip}, "
                   f"image_size={image_size}, hm_size={hm_size}")

    def _enumerate_clips(self) -> List[Path]:
        """Enumerate all clip directories in the split."""
        split_dir = self.root / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        clips = []

        # Find all scene directories
        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]

        for scene_dir in scene_dirs:
            # Find all clip directories in scene
            clip_dirs = sorted([d for d in scene_dir.iterdir()
                              if d.is_dir() and d.name.startswith('clip_')])
            clips.extend(clip_dirs)

        if len(clips) == 0:
            logger.warning(f"No clips found in {split_dir}")

        return clips

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Load and return one training sample.

        Returns:
            dict: {
                "frames": Tensor[T, 3, H, W] - RGB frames
                "text": None - No text instruction for this dataset
                "gt_heatmaps": Tensor[K, Hm, Wm] - Ground truth heatmaps
                "mask": Tensor[K] - Validity mask for heatmaps
                "meta": Dict - Metadata
            }
        """
        clip_dir = self.clips[idx]

        try:
            # Load frames
            frames = self._load_frames(clip_dir)

            # Load heatmaps and mask
            gt_heatmaps, mask = self._load_heatmaps(clip_dir)

            # Load metadata
            meta = self._load_metadata(clip_dir)

            # Ensure correct tensor shapes
            frames = self._validate_frames_shape(frames)
            gt_heatmaps, mask = self._validate_heatmaps_shape(gt_heatmaps, mask)

            return {
                "frames": frames,           # [T, 3, H, W]
                "text": "",                # Empty string instead of None
                "gt_heatmaps": gt_heatmaps, # [K, Hm, Wm]
                "mask": mask,              # [K]
                "meta": meta               # Dict
            }

        except Exception as e:
            logger.error(f"Error loading clip {clip_dir}: {e}")
            # Return dummy data to avoid crash
            return self._get_dummy_sample()

    def _load_frames(self, clip_dir: Path) -> torch.Tensor:
        """Load RGB frames from clip directory."""
        rgb_dir = clip_dir / "rgb"

        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

        # Find all RGB files
        rgb_files = sorted(rgb_dir.glob("*.png"))

        if len(rgb_files) == 0:
            raise FileNotFoundError(f"No RGB files found in {rgb_dir}")

        # Load frames
        frames = []
        target_w, target_h = self.image_size

        for rgb_file in rgb_files[:self.frames_per_clip]:
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
        frames_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]

        return frames_tensor

    def _load_heatmaps(self, clip_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load heatmaps and mask from clip directory."""
        heatmaps_file = clip_dir / "heatmaps.npy"
        mask_file = clip_dir / "mask.npy"

        if not heatmaps_file.exists():
            raise FileNotFoundError(f"Heatmaps file not found: {heatmaps_file}")
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_file}")

        # Load heatmaps and mask
        heatmaps = np.load(heatmaps_file).astype(np.float32)  # [K, Hm_orig, Wm_orig]
        mask = np.load(mask_file).astype(np.float32)          # [K]

        # Convert to tensors
        heatmaps_tensor = torch.from_numpy(heatmaps)
        mask_tensor = torch.from_numpy(mask)

        # Resize heatmaps if needed
        K, Hm_orig, Wm_orig = heatmaps_tensor.shape
        target_w, target_h = self.hm_size

        if (Hm_orig, Wm_orig) != (target_h, target_w):
            # Resize each heatmap individually and renormalize
            resized_heatmaps = []

            for k in range(K):
                hm = heatmaps_tensor[k:k+1].unsqueeze(0)  # [1, 1, Hm_orig, Wm_orig]

                if mask_tensor[k] > 0.5:  # Valid heatmap
                    # Resize using bilinear interpolation
                    hm_resized = F.interpolate(hm, size=(target_h, target_w),
                                             mode='bilinear', align_corners=False)
                    hm_resized = hm_resized.squeeze(0).squeeze(0)  # [target_h, target_w]

                    # Renormalize to ensure sum=1
                    hm_sum = hm_resized.sum()
                    if hm_sum > 0:
                        hm_resized = hm_resized / hm_sum
                else:
                    # Invalid heatmap - keep as zeros
                    hm_resized = torch.zeros(target_h, target_w)

                resized_heatmaps.append(hm_resized)

            heatmaps_tensor = torch.stack(resized_heatmaps, dim=0)

        return heatmaps_tensor, mask_tensor

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

    def _validate_frames_shape(self, frames: torch.Tensor) -> torch.Tensor:
        """Validate and adjust frames tensor shape."""
        T, C, H, W = frames.shape

        if T != self.frames_per_clip:
            logger.warning(f"Frame count mismatch: expected {self.frames_per_clip}, got {T}")

            if T > self.frames_per_clip:
                # Truncate extra frames
                frames = frames[:self.frames_per_clip]
            else:
                # Pad with last frame
                last_frame = frames[-1:].repeat(self.frames_per_clip - T, 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)

        return frames

    def _validate_heatmaps_shape(self, heatmaps: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and adjust heatmaps and mask tensor shapes."""
        K = heatmaps.shape[0]

        if K != self.heatmap_per_clip:
            logger.warning(f"Heatmap count mismatch: expected {self.heatmap_per_clip}, got {K}")

            if K > self.heatmap_per_clip:
                # Truncate extra heatmaps
                heatmaps = heatmaps[:self.heatmap_per_clip]
                mask = mask[:self.heatmap_per_clip]
            else:
                # Pad with zero heatmaps
                target_h, target_w = self.hm_size
                zero_heatmaps = torch.zeros(self.heatmap_per_clip - K, target_h, target_w)
                zero_mask = torch.zeros(self.heatmap_per_clip - K)

                heatmaps = torch.cat([heatmaps, zero_heatmaps], dim=0)
                mask = torch.cat([mask, zero_mask], dim=0)

        return heatmaps, mask

    def _get_dummy_sample(self) -> Dict[str, Union[torch.Tensor, Dict]]:
        """Generate dummy sample for error cases."""
        target_w, target_h = self.image_size
        hm_w, hm_h = self.hm_size

        return {
            "frames": torch.zeros(self.frames_per_clip, 3, target_h, target_w),
            "text": "",
            "gt_heatmaps": torch.zeros(self.heatmap_per_clip, hm_h, hm_w),
            "mask": torch.zeros(self.heatmap_per_clip),
            "meta": {"error": "Failed to load clip"}
        }


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