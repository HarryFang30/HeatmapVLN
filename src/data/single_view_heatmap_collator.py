"""Worker-side collation for front-view-only heatmap training.

The input contract is deliberately narrower than the panoramic collator:

* each sample contributes ``K`` historical *front* RGB observations followed
  by the current *front* RGB observation;
* images are flattened in sample-major order and passed through the exact
  ``AutoProcessor.image_processor`` path used by Qwen2.5-VL for image inputs;
* no raw RGB tensors and no panoramic RGB keys leave the worker;
* four-view heatmap/visibility supervision remains occurrence-aligned with the
  ``K`` historical observations.

The model can recover per-sample image spans from ``num_histories``: sample
``b`` owns ``num_histories[b] + 1`` consecutive entries in
``image_grid_thw``; its last entry is the current observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


_FORBIDDEN_PANORAMIC_RGB_KEYS = frozenset({"current_views", "history_panoramas"})
_DIRECTION_ORDER = ("front", "right", "back", "left")
_POSE_CONVENTION = "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"


def _to_pil_rgb(image: torch.Tensor | np.ndarray | Image.Image) -> Image.Image:
    """Convert one CHW/HWC image to RGB PIL without changing its geometry."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if torch.is_tensor(image):
        value = image.detach().cpu()
        if value.ndim != 3:
            raise ValueError(f"Expected a rank-3 image tensor, got {tuple(value.shape)}")
        if value.shape[0] in (1, 3):
            value = value.permute(1, 2, 0)
        array = value.numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    if array.ndim != 3 or array.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC image with 1 or 3 channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array, mode="RGB")


def _pad_first_dim(tensors: list[torch.Tensor], *, pad_value: float = 0.0) -> torch.Tensor:
    """Pad variable-length history tensors along dimension zero, then stack."""
    if not tensors:
        raise ValueError("Cannot collate an empty tensor list")
    max_length = max(int(value.shape[0]) for value in tensors)
    output = []
    for value in tensors:
        if value.shape[0] < max_length:
            padding = torch.full(
                (max_length - value.shape[0], *value.shape[1:]),
                fill_value=pad_value,
                dtype=value.dtype,
            )
            value = torch.cat((value, padding), dim=0)
        output.append(value)
    return torch.stack(output, dim=0)


class SingleViewHeatmapCollator:
    """Preprocess front-view images in DataLoader workers.

    Args:
        processor: The ``AutoProcessor`` loaded from the native InternNav model.
          This collator intentionally calls ``processor.image_processor`` rather
          than the chat/video processor, so every history/current observation is
          represented by one independent ``image_grid_thw`` row.
        require_four_view_targets: Fail closed unless every sample contains a
          ``[K, 4, H, W]`` heatmap and optional ``[K, 4]`` visibility target.
        retain_text: Retain instructions for diagnostics. The heatmap-only model
          path does not need tokenized text.
    """

    def __init__(
        self,
        processor: Any,
        *,
        require_four_view_targets: bool = True,
        retain_text: bool = True,
    ) -> None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None or not callable(image_processor):
            raise TypeError(
                "SingleViewHeatmapCollator requires an InternNav AutoProcessor "
                "with a callable image_processor"
            )
        self.image_processor = image_processor
        self.require_four_view_targets = bool(require_four_view_targets)
        self.retain_text = bool(retain_text)

    @staticmethod
    def _validate_sample(sample: dict[str, Any], sample_index: int) -> int:
        leaked = sorted(_FORBIDDEN_PANORAMIC_RGB_KEYS.intersection(sample))
        if leaked:
            raise RuntimeError(
                f"sample[{sample_index}] contains forbidden panoramic RGB keys: {leaked}. "
                "Set data.sliding_window.single_view_rgb_input=true."
            )

        history = sample.get("history_frames")
        current = sample.get("current_frame")
        if not torch.is_tensor(history) or history.ndim != 4 or history.shape[1] != 3:
            raise ValueError(
                f"sample[{sample_index}].history_frames must be [K,3,H,W], "
                f"got {getattr(history, 'shape', None)}"
            )
        if history.shape[0] < 1:
            raise ValueError(f"sample[{sample_index}] has no historical observation")
        if not torch.is_tensor(current) or current.ndim != 3 or current.shape[0] != 3:
            raise ValueError(
                f"sample[{sample_index}].current_frame must be [3,H,W], "
                f"got {getattr(current, 'shape', None)}"
            )

        num_history = int(history.shape[0])
        for key in ("history_rel_poses", "gt_visibility"):
            value = sample.get(key)
            if torch.is_tensor(value) and int(value.shape[0]) != num_history:
                raise ValueError(
                    f"sample[{sample_index}].{key} has {value.shape[0]} histories; "
                    f"RGB input has {num_history}"
                )
        heatmap = sample.get("heatmap")
        if torch.is_tensor(heatmap) and heatmap.ndim >= 4 and heatmap.shape[0] != num_history:
            raise ValueError(
                f"sample[{sample_index}].heatmap has {heatmap.shape[0]} histories; "
                f"RGB input has {num_history}"
            )
        return num_history

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("SingleViewHeatmapCollator received an empty batch")

        # ShmBypassDataset represents tensors as NumPy arrays. Normalize a
        # shallow copy so this collator has one invariant in both IPC modes.
        batch = [
            {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in sample.items()
            }
            for sample in batch
        ]

        num_histories = [
            self._validate_sample(sample, index)
            for index, sample in enumerate(batch)
        ]
        for index, sample in enumerate(batch):
            if tuple(sample.get("heatmap_direction_order", ())) != _DIRECTION_ORDER:
                raise ValueError(
                    f"sample[{index}] heatmap direction order must be {_DIRECTION_ORDER}"
                )
            if sample.get("history_pose_convention") != _POSE_CONVENTION:
                raise ValueError(
                    f"sample[{index}] history pose convention must be {_POSE_CONVENTION}"
                )
        providers = [sample.get("history_pose_provider") for sample in batch]
        if any(provider is None for provider in providers):
            raise ValueError(
                "Every single-view sample must declare history_pose_provider"
            )
        if len(set(providers)) != 1:
            raise ValueError(
                f"Mixed history_pose_provider values are forbidden: {providers}"
            )
        if self.require_four_view_targets:
            for index, (sample, length) in enumerate(zip(batch, num_histories)):
                heatmap = sample.get("heatmap")
                if not torch.is_tensor(heatmap) or heatmap.ndim != 4:
                    raise ValueError(
                        f"sample[{index}].heatmap must be [K,4,H,W], "
                        f"got {getattr(heatmap, 'shape', None)}"
                    )
                if tuple(heatmap.shape[:2]) != (length, 4):
                    raise ValueError(
                        f"sample[{index}].heatmap must begin with [{length},4], "
                        f"got {tuple(heatmap.shape)}"
                    )
                visibility = sample.get("gt_visibility")
                if visibility is not None and tuple(visibility.shape) != (length, 4):
                    raise ValueError(
                        f"sample[{index}].gt_visibility must be [{length},4], "
                        f"got {tuple(visibility.shape)}"
                    )

        flat_images: list[Image.Image] = []
        for sample in batch:
            flat_images.extend(_to_pil_rgb(frame) for frame in sample["history_frames"])
            flat_images.append(_to_pil_rgb(sample["current_frame"]))

        with torch.no_grad():
            encoded = self.image_processor(images=flat_images, return_tensors="pt")
        if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
            raise RuntimeError(
                "InternNav image processor did not return pixel_values and image_grid_thw"
            )
        pixel_values = encoded["pixel_values"]
        image_grid_thw = encoded["image_grid_thw"]
        expected_images = sum(length + 1 for length in num_histories)
        if image_grid_thw.ndim != 2 or tuple(image_grid_thw.shape[1:]) != (3,):
            raise RuntimeError(
                f"image_grid_thw must be [num_images,3], got {tuple(image_grid_thw.shape)}"
            )
        if int(image_grid_thw.shape[0]) != expected_images:
            raise RuntimeError(
                f"Image processor returned {image_grid_thw.shape[0]} grids for "
                f"{expected_images} flattened images"
            )

        lengths = torch.tensor(num_histories, dtype=torch.long)
        max_history = int(lengths.max().item())
        history_mask = (
            torch.arange(max_history, dtype=torch.long).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        image_counts = lengths + 1
        image_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), image_counts.cumsum(dim=0)),
            dim=0,
        )

        result: dict[str, Any] = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "num_histories": lengths,
            "image_offsets": image_offsets,
            "history_mask": history_mask,
            "heatmap_direction_order": _DIRECTION_ORDER,
            "history_pose_convention": _POSE_CONVENTION,
            "history_pose_provider": providers[0],
            "heatmap": _pad_first_dim([sample["heatmap"] for sample in batch]),
            "action": torch.stack([sample["action"] for sample in batch], dim=0),
            "action_valid": torch.tensor([sample["action_valid"] for sample in batch]),
            "discrete_action": torch.tensor(
                [sample.get("discrete_action", 1) for sample in batch]
            ),
            "is_stop": torch.tensor([sample.get("is_stop", 0.0) for sample in batch]),
        }
        if self.retain_text:
            result["text"] = [sample.get("text", "") for sample in batch]
        if any("sample_identity" in sample for sample in batch):
            if not all("sample_identity" in sample for sample in batch):
                raise ValueError(
                    "sample_identity must be present for every sample or none"
                )
            identities = [str(sample["sample_identity"]) for sample in batch]
            if any(not value for value in identities):
                raise ValueError("sample_identity values must be non-empty")
            result["sample_identity"] = identities

        for key in ("gt_visibility", "history_rel_poses", "history_poses"):
            if key in batch[0]:
                if not all(key in sample for sample in batch):
                    raise ValueError(f"Optional key {key!r} is missing from part of the batch")
                result[key] = _pad_first_dim([sample[key] for sample in batch])
        for key in ("current_pose", "current_depth", "intrinsics"):
            if key in batch[0]:
                if not all(key in sample for sample in batch):
                    raise ValueError(f"Optional key {key!r} is missing from part of the batch")
                result[key] = torch.stack([sample[key] for sample in batch], dim=0)
        if "is_flipped" in batch[0]:
            result["is_flipped"] = torch.tensor(
                [sample.get("is_flipped", False) for sample in batch],
                dtype=torch.bool,
            )

        # Fail closed: raw RGB and panoramic tensors must never cross the
        # DataLoader worker boundary in this path.
        assert not _FORBIDDEN_PANORAMIC_RGB_KEYS.intersection(result)
        assert "history_frames" not in result and "current_frame" not in result
        return result
