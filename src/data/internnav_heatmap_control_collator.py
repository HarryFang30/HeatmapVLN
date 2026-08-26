"""Joint worker-side collation for frozen InternNav + heatmap control.

The batch intentionally contains two independent processor products:

* pano_inputs is the released InternNav System-2 protocol
  (independent valid front-history images, current front view, lookdown image,
  and TRAJ tokens).
* heatmap_single_view_inputs contains independent still images for the
  frozen single-view heatmap feature extractor, including fixed history slots.

Keeping these namespaces separate prevents the heatmap image grouping from
silently changing the native System-2 prompt.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .amb3r_pose_cache import AMB3R_POSE_PROVIDER
from .future_trajectory_batch import (
    assert_no_future_teacher_inputs,
    stack_future_trajectory_targets,
)
from .panoramic_tokenized_collator import PanoramicTokenizedCollator
from .single_view_heatmap_collator import _to_pil_rgb


_DIRECTION_ORDER = ("front", "right", "back", "left")
_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)


def _tensorize_shallow(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        for key, value in sample.items()
    }


class InternNavHeatmapControlCollator:
    """Build native System-2 and frozen heatmap inputs from the same sample."""

    def __init__(
        self,
        processor: Any,
        *,
        n_traj_query: int = 4,
        max_seq_length: int = 8192,
        teacher_force_system2_answer: bool = True,
        include_future_trajectory_targets: bool = False,
        required_history_pose_provider: str | None = None,
    ) -> None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None or not callable(image_processor):
            raise TypeError(
                "InternNavHeatmapControlCollator requires an AutoProcessor "
                "with a callable image_processor"
            )
        if int(n_traj_query) <= 0:
            raise ValueError("n_traj_query must be positive for System-1 training")
        self.image_processor = image_processor
        self.include_future_trajectory_targets = bool(
            include_future_trajectory_targets
        )
        if required_history_pose_provider not in (None, AMB3R_POSE_PROVIDER):
            raise ValueError(
                "required_history_pose_provider must be None or "
                f"{AMB3R_POSE_PROVIDER!r}"
            )
        self.required_history_pose_provider = required_history_pose_provider
        self.native_collator = PanoramicTokenizedCollator(
            processor,
            n_traj_query=int(n_traj_query),
            sft_mode=bool(teacher_force_system2_answer),
            sft_protocol="internnav",
            build_sft_labels=False,
            max_seq_length=int(max_seq_length),
            include_heatmap_targets=False,
            include_history_rel_poses=True,
            retain_raw_panoramic_views=False,
            compute_pano_text_anchor_positions=False,
            heatmap_layout=False,
            force_internnav_prompt=True,
        )

    @staticmethod
    def _front_history(sample: dict[str, Any]) -> torch.Tensor:
        value = sample.get("history_frames")
        if value is None:
            panoramas = sample.get("history_panoramas")
            if torch.is_tensor(panoramas) and panoramas.ndim == 5:
                value = panoramas[:, 0]
        if not torch.is_tensor(value) or value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(
                "history front RGB must be [K,3,H,W] (history_frames or "
                "history_panoramas[:,0])"
            )
        if value.shape[0] < 1:
            raise ValueError("heatmap control requires at least one history slot")
        return value

    @staticmethod
    def _front_current(sample: dict[str, Any]) -> torch.Tensor:
        value = sample.get("current_frame")
        if value is None:
            views = sample.get("current_views")
            if torch.is_tensor(views) and views.ndim == 4:
                value = views[0]
        if not torch.is_tensor(value) or value.ndim != 3 or value.shape[0] != 3:
            raise ValueError(
                "current front RGB must be [3,H,W] (current_frame or "
                "current_views[0])"
            )
        return value

    @staticmethod
    def _canonicalize_trajectory(sample: dict[str, Any]) -> None:
        trajectory = sample.get("trajectory")
        if not torch.is_tensor(trajectory):
            raise ValueError("heatmap control samples require trajectory")
        trajectory_was_sequence = trajectory.ndim == 3
        if trajectory.ndim == 3:
            trajectory = trajectory[0]
        if trajectory.ndim != 2 or trajectory.shape[-1] != 3:
            raise ValueError(
                f"trajectory must canonicalize to [T,3], got {tuple(trajectory.shape)}"
            )
        sample["trajectory"] = trajectory

        valid = sample.get("trajectory_valid", 1.0)
        if torch.is_tensor(valid):
            valid = valid.reshape(-1)[0]
        sample["trajectory_valid"] = valid

        images = sample.get("traj_images")
        if not torch.is_tensor(images):
            raise ValueError("heatmap control samples require traj_images")
        source = str(sample.get("source_type", ""))
        if images.ndim == 3:
            images = torch.stack((images, images), dim=0)
        elif images.ndim == 4 and (
            source == "expert" or trajectory_was_sequence
        ):
            images = torch.stack((images[0], images[0]), dim=0)
        if (
            images.ndim != 4
            or images.shape[0] != 2
            or images.shape[-1] != 3
        ):
            raise ValueError(
                "traj_images must canonicalize to [2,H,W,3]; "
                f"source={source!r}, got {tuple(images.shape)}"
            )
        sample["traj_images"] = images

    def _canonicalize_sample(
        self,
        raw_sample: dict[str, Any],
        sample_index: int,
    ) -> dict[str, Any]:
        sample = _tensorize_shallow(dict(raw_sample))
        history = self._front_history(sample)
        current = self._front_current(sample)
        sample["history_frames"] = history
        sample["current_frame"] = current
        history_count = int(history.shape[0])

        if sample.get("lookdown_frame") is None:
            raise ValueError(
                f"sample[{sample_index}] lacks lookdown_frame required by native System-2"
            )
        if tuple(sample.get("heatmap_direction_order", ())) != _DIRECTION_ORDER:
            raise ValueError(
                f"sample[{sample_index}] direction order must be {_DIRECTION_ORDER}"
            )
        if sample.get("history_pose_convention") != _POSE_CONVENTION:
            raise ValueError(
                f"sample[{sample_index}] pose convention must be {_POSE_CONVENTION}"
            )
        provider = sample.get("history_pose_provider")
        if (
            self.required_history_pose_provider is not None
            and provider != self.required_history_pose_provider
        ):
            raise ValueError(
                f"sample[{sample_index}] history_pose_provider must be "
                f"{self.required_history_pose_provider!r}, got {provider!r}; "
                "GT fallback/mixed pose domains are forbidden"
            )

        rel_poses = sample.get("history_rel_poses")
        ages = sample.get("history_age_steps")
        if (
            not torch.is_tensor(rel_poses)
            or tuple(rel_poses.shape) != (history_count, 4)
        ):
            raise ValueError(
                f"sample[{sample_index}].history_rel_poses must be "
                f"[{history_count},4]"
            )
        if not torch.is_tensor(ages) or tuple(ages.shape) != (history_count,):
            raise ValueError(
                f"sample[{sample_index}].history_age_steps must be [{history_count}]"
            )
        valid = sample.get("history_valid_mask")
        if valid is None:
            valid = sample.get("history_mask")
        if not torch.is_tensor(valid) or tuple(valid.shape) != (history_count,):
            raise ValueError(
                f"sample[{sample_index}].history_valid_mask must be [{history_count}]"
            )
        sample["history_valid_mask"] = valid.bool()
        sample["history_mask"] = valid.float()
        sample["history_age_steps"] = ages.to(dtype=torch.long)
        self._canonicalize_trajectory(sample)
        return sample

    def _encode_heatmap_images(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        num_histories = [
            int(sample["history_frames"].shape[0])
            for sample in samples
        ]
        flat_images = []
        for sample in samples:
            flat_images.extend(
                _to_pil_rgb(frame)
                for frame in sample["history_frames"]
            )
            flat_images.append(_to_pil_rgb(sample["current_frame"]))

        with torch.no_grad():
            encoded = self.image_processor(
                images=flat_images,
                return_tensors="pt",
            )
        required = {"pixel_values", "image_grid_thw"}
        missing = required - set(encoded)
        if missing:
            raise RuntimeError(
                f"InternNav image processor omitted {sorted(missing)}"
            )
        expected = sum(count + 1 for count in num_histories)
        grid = encoded["image_grid_thw"]
        if grid.ndim != 2 or grid.shape[1] != 3 or grid.shape[0] != expected:
            raise RuntimeError(
                "heatmap image_grid_thw does not match flattened independent "
                f"images: expected [{expected},3], got {tuple(grid.shape)}"
            )
        return {
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": grid,
        }, num_histories

    @staticmethod
    def _compact_native_history(sample: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Select only valid observations for the released System-2 prompt.

        DAgger records use fixed history slots and may contain non-prefix masks.
        Those slots are meaningful to the heatmap memory branch, but padding
        observations must never become image placeholders in native InternNav.
        """
        valid = sample["history_valid_mask"].bool()
        count = int(valid.sum().item())
        compact = dict(sample)
        history_length = int(valid.shape[0])
        for key in (
            "history_frames",
            "history_rel_poses",
            "history_poses",
            "history_frame_ids",
            "history_age_steps",
            "gt_visibility",
        ):
            value = sample.get(key)
            if (
                torch.is_tensor(value)
                and value.ndim >= 1
                and int(value.shape[0]) == history_length
            ):
                compact[key] = value[valid]
        compact["history_valid_mask"] = torch.ones(
            count, dtype=torch.bool, device=valid.device
        )
        compact["history_mask"] = torch.ones(
            count, dtype=torch.float32, device=valid.device
        )
        return compact, count

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("InternNavHeatmapControlCollator received an empty batch")
        samples = [
            self._canonicalize_sample(sample, index)
            for index, sample in enumerate(batch)
        ]
        future_targets = (
            stack_future_trajectory_targets(samples)
            if self.include_future_trajectory_targets
            else None
        )
        heatmap_inputs, num_histories = self._encode_heatmap_images(samples)
        native_samples = []
        native_num_histories = []
        for sample in samples:
            native_sample, native_count = self._compact_native_history(sample)
            native_samples.append(native_sample)
            native_num_histories.append(native_count)

        # The native collator clears its input dictionaries to release RGB
        # memory early.  It receives compact valid history; the heatmap branch
        # retains the original fixed-K history and explicit validity mask.
        result = self.native_collator(native_samples)
        fixed_history = self.native_collator._stack_padded_history_frames(samples)
        result["history_frames"] = fixed_history["history_frames"]
        for key in (
            "history_rel_poses",
            "history_poses",
            "history_frame_ids",
            "history_age_steps",
        ):
            if all(key in sample for sample in samples):
                result[key] = self.native_collator._stack_padded_first_dim(
                    samples, key
                )
        result["history_valid_mask"] = (
            self.native_collator._stack_padded_first_dim(
                samples, "history_valid_mask"
            ).bool()
        )
        result["history_mask"] = result["history_valid_mask"].float()
        # ``native_collator`` is intentionally configured with
        # include_heatmap_targets=False because it consumes compacted
        # System-2 history.  History supervision, however, belongs to the
        # original fixed-K heatmap slots.  Restore both target tensors from
        # ``samples`` before the native collator clears its private copies.
        # This keeps Past loss aligned with history_valid_mask.  Batches that
        # carry no fixed-K targets simply pass through without them.
        if all("heatmap" in sample for sample in samples):
            result["heatmap"] = self.native_collator._stack_padded_first_dim(
                samples, "heatmap"
            )
        if all("gt_visibility" in sample for sample in samples):
            result["gt_visibility"] = (
                self.native_collator._stack_padded_first_dim(
                    samples, "gt_visibility"
                )
            )
        result["heatmap_single_view_inputs"] = heatmap_inputs
        result["heatmap_single_view_num_histories"] = num_histories
        result["heatmap_control_history_mask"] = result["history_valid_mask"]
        result["native_system2_num_histories"] = native_num_histories
        identities = [sample.get("sample_identity") for sample in samples]
        if all(identity is not None and str(identity) != "" for identity in identities):
            result["sample_identity"] = [str(identity) for identity in identities]
        if self.required_history_pose_provider is not None:
            result["history_pose_provider"] = [
                self.required_history_pose_provider for _ in samples
            ]
            # In AMB3R mode the GT c2w/depth tensors are target-build inputs,
            # never model inputs.  Dataset-side labels are already complete.
            for teacher_key in (
                "current_pose",
                "current_camera_pose",
                "current_agent_pose",
                "history_poses",
                "current_depth",
            ):
                result.pop(teacher_key, None)
        if future_targets is not None:
            result.update(future_targets)
            assert_no_future_teacher_inputs(result)
        return result
