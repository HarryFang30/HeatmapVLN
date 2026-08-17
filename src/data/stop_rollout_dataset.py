"""Multimodal on-policy examples for System2 navigation-LoRA continuation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.models.heatmap.input_constructor import parse_structured_pano_output

STOP_MULTIMODAL_EXAMPLE_SCHEMA = "heatmapvln-system2-stop-multimodal-example-v1"
_VIEW_NAMES = ("front", "right", "back", "left")


def _load_rgb_tensor(path: Path, image_size: tuple[int, int]) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _best_non_stop_target(record: dict[str, Any]) -> tuple[str, list[int] | None, str]:
    output = str(record.get("effective_output") or record.get("original_output") or "")
    parsed = parse_structured_pano_output(output, image_size=None)
    if parsed.kind == "pixel" and parsed.view_id in _VIEW_NAMES and parsed.pixel_goal:
        return parsed.view_id, [int(value) for value in parsed.pixel_goal], "pixel"
    if parsed.kind in {"turn", "turn_left", "turn_right"}:
        return "view_turn", None, "turn"

    scores = record.get("system2_decision_scores")
    probabilities = scores.get("class_probabilities", {}) if isinstance(scores, dict) else {}
    candidates = {
        name: float(probabilities.get(name, float("-inf")))
        for name in (*_VIEW_NAMES, "turn")
    }
    selected = max(candidates, key=candidates.get)
    if selected == "turn":
        return "view_turn", None, "turn"
    image_size = record.get("image_size") or [256, 256]
    return selected, [int(image_size[0]) // 2, int(image_size[1]) // 2], "pixel"


class System2StopMultimodalDataset(Dataset):
    """Load privileged train-split rollout inputs without permitting eval leakage."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        *,
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        self.image_size = tuple(int(value) for value in image_size)
        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError(f"Invalid rollout image size: {self.image_size}")

        records: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        resolved_roots: list[Path] = []
        for raw_root in roots:
            root = Path(raw_root).expanduser().resolve()
            labels_path = root / "system2_stop_multimodal_examples.jsonl"
            if not labels_path.is_file():
                raise FileNotFoundError(f"Missing multimodal STOP labels: {labels_path}")
            resolved_roots.append(root)
            with labels_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("schema") != STOP_MULTIMODAL_EXAMPLE_SCHEMA:
                        raise ValueError(
                            f"Unexpected STOP example schema at {labels_path}:{line_number}: "
                            f"{record.get('schema')!r}"
                        )
                    if record.get("dataset_split") != "train":
                        raise ValueError(
                            "Refusing non-train STOP rollout data: "
                            f"split={record.get('dataset_split')!r} "
                            f"key={record.get('key')!r}"
                        )
                    target = record.get("stop_target")
                    if target not in (0, 1):
                        continue
                    if not isinstance(record.get("original_terminal"), bool):
                        raise ValueError(
                            "STOP rollout example is missing the recorded original "
                            f"terminal decision: key={record.get('key')!r}"
                        )
                    key = str(record.get("key") or "")
                    if not key or key in seen_keys:
                        raise ValueError(f"Missing or duplicate STOP rollout key: {key!r}")
                    seen_keys.add(key)
                    normalized = dict(record)
                    normalized["_root"] = root
                    records.append(normalized)

        if not records:
            raise RuntimeError("No labelled multimodal STOP rollout examples were found")
        self.records = records
        self.roots = tuple(resolved_roots)
        self.targets = tuple(int(record["stop_target"]) for record in records)
        self.original_terminals = tuple(
            bool(record["original_terminal"]) for record in records
        )
        self.sample_scene_ids = tuple(str(record["scene_id"]) for record in records)
        if len(set(self.sample_scene_ids)) < 2:
            raise RuntimeError("Multimodal STOP rollout data requires at least two scenes")
        if not any(self.targets) or all(self.targets):
            raise RuntimeError("Multimodal STOP rollout data requires STOP and non-STOP rows")

    @classmethod
    def _from_records(
        cls,
        parent: System2StopMultimodalDataset,
        indices: Iterable[int],
    ) -> System2StopMultimodalDataset:
        dataset = object.__new__(cls)
        dataset.image_size = parent.image_size
        dataset.roots = parent.roots
        dataset.records = [parent.records[int(index)] for index in indices]
        dataset.targets = tuple(int(record["stop_target"]) for record in dataset.records)
        dataset.original_terminals = tuple(
            bool(record["original_terminal"]) for record in dataset.records
        )
        dataset.sample_scene_ids = tuple(str(record["scene_id"]) for record in dataset.records)
        return dataset

    def subset_by_indices(self, indices: Iterable[int]) -> System2StopMultimodalDataset:
        return self._from_records(self, indices)

    def split_by_scene(
        self,
        *,
        holdout_fraction: float,
        seed: int,
    ) -> tuple[System2StopMultimodalDataset, System2StopMultimodalDataset]:
        """Return deterministic, scene-disjoint train and validation views."""
        if not 0.0 < float(holdout_fraction) < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        scene_to_indices: dict[str, list[int]] = {}
        for index, scene_id in enumerate(self.sample_scene_ids):
            scene_to_indices.setdefault(str(scene_id), []).append(index)
        scenes = sorted(scene_to_indices)
        if len(scenes) < 2:
            raise RuntimeError("Scene holdout requires at least two scenes")
        ordered = sorted(
            scenes,
            key=lambda scene_id: hashlib.sha256(
                f"{int(seed)}:{scene_id}".encode()
            ).digest(),
        )
        holdout_count = min(
            len(scenes) - 1,
            max(1, round(len(scenes) * float(holdout_fraction))),
        )
        train_indices = [
            index
            for scene_id in ordered[holdout_count:]
            for index in scene_to_indices[scene_id]
        ]
        validation_indices = [
            index
            for scene_id in ordered[:holdout_count]
            for index in scene_to_indices[scene_id]
        ]
        train = self.subset_by_indices(train_indices)
        validation = self.subset_by_indices(validation_indices)
        if set(train.sample_scene_ids) & set(validation.sample_scene_ids):
            raise RuntimeError("System2 rollout train/validation scenes overlap")
        for split_name, split in (("train", train), ("validation", validation)):
            if not any(split.targets) or all(split.targets):
                raise RuntimeError(
                    f"System2 rollout {split_name} split requires STOP and non-STOP rows"
                )
        return train, validation

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _resolve_views(root: Path, views: Any) -> dict[str, Path]:
        if not isinstance(views, dict):
            raise ValueError("STOP rollout example is missing a view mapping")
        result: dict[str, Path] = {}
        for name in _VIEW_NAMES:
            raw_path = views.get(name)
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"STOP rollout example is missing {name!r} image")
            path = Path(raw_path)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError(f"Missing STOP rollout image: {path}")
            result[name] = path
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        root = record["_root"]
        current_paths = self._resolve_views(root, record.get("current_views"))
        current_views = torch.stack(
            [_load_rgb_tensor(current_paths[name], self.image_size) for name in _VIEW_NAMES]
        )

        history_tensors = []
        for history_views in record.get("history_views") or []:
            paths = self._resolve_views(root, history_views)
            history_tensors.append(
                torch.stack(
                    [_load_rgb_tensor(paths[name], self.image_size) for name in _VIEW_NAMES]
                )
            )
        history_panoramas = (
            torch.stack(history_tensors)
            if history_tensors
            else torch.empty((0, 4, 3, self.image_size[1], self.image_size[0]))
        )

        target = int(record["stop_target"])
        original_terminal = bool(record["original_terminal"])
        if target:
            pano_view_id: str | None = "view_stop"
            pano_pixel_goal = None
            pano_sample_kind = "stop"
        elif original_terminal:
            # A false STOP only tells us which token to reject. It does not
            # provide a trustworthy counterfactual waypoint, so expose the
            # rejected STOP text and let the trainer apply token unlikelihood.
            pano_view_id = "view_stop"
            pano_pixel_goal = None
            pano_sample_kind = "stop_reject"
        else:
            pano_view_id, pano_pixel_goal, pano_sample_kind = _best_non_stop_target(record)

        return {
            "history_frames": current_views[:1].clone(),
            "current_frame": current_views[0],
            "action": torch.zeros(2, dtype=torch.float32),
            "action_valid": 0.0,
            "discrete_action": 0 if target else 1,
            "is_stop": float(target),
            "text": str(record["instruction"]),
            "current_views": current_views,
            "history_panoramas": history_panoramas,
            "pano_view_id": pano_view_id,
            "pano_pixel_goal": pano_pixel_goal,
            "pano_sample_kind": pano_sample_kind,
            "stop_rollout_key": str(record["key"]),
            "system2_original_terminal": original_terminal,
            "system2_oracle_stop_target": target,
        }


class MixedSystem2SFTDataset(Dataset):
    """Deterministically mix native SFT replay with on-policy corrections.

    The virtual epoch covers every native sample once. On-policy STOP and
    regular non-STOP rows, and false-STOP rejection rows are cycled through
    separate slots. This keeps the rare failure mode visible without inventing
    a counterfactual waypoint or adding an inference-time policy.
    """

    def __init__(
        self,
        native_dataset: Dataset,
        rollout_dataset: System2StopMultimodalDataset,
        *,
        native_slots: int = 14,
        positive_slots: int = 3,
        regular_negative_slots: int = 1,
        false_stop_negative_slots: int = 2,
        regular_negative_min_stop_log_odds: float | None = None,
        pair_false_stops: bool = False,
    ) -> None:
        slot_counts = {
            "native": int(native_slots),
            "onpolicy_positive": int(positive_slots),
            "onpolicy_regular_negative": int(regular_negative_slots),
            "onpolicy_false_stop_negative": int(false_stop_negative_slots),
        }
        if any(count <= 0 for count in slot_counts.values()):
            raise ValueError(f"Mixed System2 SFT slots must be positive: {slot_counts}")
        if len(native_dataset) <= 0:
            raise RuntimeError("Native System2 SFT replay dataset is empty")

        positive_indices = [
            index
            for index, target in enumerate(rollout_dataset.targets)
            if int(target) == 1
        ]
        all_regular_negative_indices = [
            index
            for index, (target, terminal) in enumerate(
                zip(rollout_dataset.targets, rollout_dataset.original_terminals)
            )
            if int(target) == 0 and not bool(terminal)
        ]
        regular_negative_threshold = (
            None
            if regular_negative_min_stop_log_odds is None
            else float(regular_negative_min_stop_log_odds)
        )
        if regular_negative_threshold is not None and not math.isfinite(
            regular_negative_threshold
        ):
            raise ValueError("Regular-negative STOP log-odds threshold must be finite")

        def _stop_log_odds(index: int) -> float:
            scores = rollout_dataset.records[index].get("system2_decision_scores")
            raw_value = scores.get("stop_log_odds") if isinstance(scores, dict) else None
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Hard regular-negative mining requires finite stop_log_odds: "
                    f"key={rollout_dataset.records[index].get('key')!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    "Hard regular-negative mining requires finite stop_log_odds: "
                    f"key={rollout_dataset.records[index].get('key')!r}"
                )
            return value

        regular_negative_indices = all_regular_negative_indices
        if regular_negative_threshold is not None:
            regular_negative_indices = [
                index
                for index in all_regular_negative_indices
                if _stop_log_odds(index) > regular_negative_threshold
            ]
        false_stop_indices = [
            index
            for index, (target, terminal) in enumerate(
                zip(rollout_dataset.targets, rollout_dataset.original_terminals)
            )
            if int(target) == 0 and bool(terminal)
        ]
        positive_by_episode: dict[tuple[str, int], list[int]] = {}
        for index in positive_indices:
            record = rollout_dataset.records[index]
            episode_key = (str(record["scene_id"]), int(record["episode_id"]))
            positive_by_episode.setdefault(episode_key, []).append(index)

        def _call_index(index: int) -> int:
            return int(rollout_dataset.records[index].get("system2_call_index") or 0)

        paired_positive_by_false: dict[int, int] = {}
        for false_index in false_stop_indices:
            false_record = rollout_dataset.records[false_index]
            episode_key = (
                str(false_record["scene_id"]),
                int(false_record["episode_id"]),
            )
            candidates = positive_by_episode.get(episode_key, [])
            if not candidates:
                continue
            false_call = _call_index(false_index)
            paired_positive_by_false[false_index] = min(
                candidates,
                key=lambda positive_index: (
                    0 if _call_index(positive_index) >= false_call else 1,
                    abs(_call_index(positive_index) - false_call),
                    str(rollout_dataset.records[positive_index]["key"]),
                ),
            )
        false_stop_candidate_count = len(false_stop_indices)
        if pair_false_stops:
            false_stop_indices = [
                index
                for index in false_stop_indices
                if index in paired_positive_by_false
            ]
        if not positive_indices or not regular_negative_indices or not false_stop_indices:
            raise RuntimeError(
                "On-policy correction data must contain STOP positives, regular "
                "negatives, and recorded false-STOP negatives"
            )

        def _stable_pool(indices: list[int], role: str) -> tuple[int, ...]:
            return tuple(
                sorted(
                    indices,
                    key=lambda index: hashlib.sha256(
                        f"{role}:{rollout_dataset.records[index]['key']}".encode()
                    ).digest(),
                )
            )

        self.native_dataset = native_dataset
        self.rollout_dataset = rollout_dataset
        self.regular_negative_candidate_count = len(all_regular_negative_indices)
        self.regular_negative_min_stop_log_odds = regular_negative_threshold
        self.positive_indices = _stable_pool(
            positive_indices, "onpolicy_positive"
        )
        self.regular_negative_indices = _stable_pool(
            regular_negative_indices, "onpolicy_regular_negative"
        )
        self.false_stop_indices = _stable_pool(
            false_stop_indices, "onpolicy_false_stop_negative"
        )
        self.pair_false_stops = bool(pair_false_stops)
        self.paired_positive_by_false = paired_positive_by_false
        self.false_stop_candidate_count = false_stop_candidate_count
        self.slot_pattern = tuple(
            role
            for role, count in slot_counts.items()
            for _ in range(count)
        )
        self._slot_ordinals: list[int] = []
        seen = {role: 0 for role in slot_counts}
        for role in self.slot_pattern:
            self._slot_ordinals.append(seen[role])
            seen[role] += 1
        self.slot_counts = slot_counts
        self.cycles = math.ceil(len(native_dataset) / slot_counts["native"])
        self._is_panoramic = bool(getattr(native_dataset, "_is_panoramic", False))
        if not self._is_panoramic:
            raise RuntimeError("Native System2 replay dataset must be panoramic")

    def __len__(self) -> int:
        return self.cycles * len(self.slot_pattern)

    def source_counts(self) -> dict[str, int]:
        return {
            role: self.cycles * count
            for role, count in self.slot_counts.items()
        }

    def pool_sizes(self) -> dict[str, int]:
        return {
            "onpolicy_positive": len(self.positive_indices),
            "onpolicy_regular_negative": len(self.regular_negative_indices),
            "onpolicy_false_stop_negative": len(self.false_stop_indices),
        }

    def regular_negative_mining_contract(self) -> dict[str, int | float | None]:
        return {
            "min_stop_log_odds": self.regular_negative_min_stop_log_odds,
            "candidate_count": self.regular_negative_candidate_count,
            "selected_count": len(self.regular_negative_indices),
        }

    def false_stop_pairing_contract(self) -> dict[str, int | bool]:
        return {
            "enabled": self.pair_false_stops,
            "candidate_count": self.false_stop_candidate_count,
            "available_paired_count": len(self.paired_positive_by_false),
            "selected_count": len(self.false_stop_indices),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        cycle, slot = divmod(int(index), len(self.slot_pattern))
        role = self.slot_pattern[slot]
        role_position = cycle * self.slot_counts[role] + self._slot_ordinals[slot]
        if role == "native":
            source_index = role_position % len(self.native_dataset)
            sample = dict(self.native_dataset[source_index])
            sample["system2_replay_role"] = "native"
            return sample

        pools = {
            "onpolicy_positive": self.positive_indices,
            "onpolicy_regular_negative": self.regular_negative_indices,
            "onpolicy_false_stop_negative": self.false_stop_indices,
        }
        pool = pools[role]
        source_index = pool[role_position % len(pool)]
        sample = dict(self.rollout_dataset[source_index])
        if role in {
            "onpolicy_regular_negative",
            "onpolicy_false_stop_negative",
        }:
            # The rollout label only proves that STOP is wrong. Do not turn the
            # model's original waypoint into a counterfactual expert target;
            # expose the STOP token so the trainer can apply unlikelihood.
            sample["pano_view_id"] = "view_stop"
            sample["pano_pixel_goal"] = None
            sample["pano_sample_kind"] = "stop_reject"
        sample["system2_replay_role"] = role
        if role == "onpolicy_false_stop_negative" and self.pair_false_stops:
            positive_index = self.paired_positive_by_false[source_index]
            pair_id = (
                f"{self.rollout_dataset.records[source_index]['key']}"
                f"::sample{role_position}"
            )
            paired_positive = dict(self.rollout_dataset[positive_index])
            paired_positive["system2_replay_role"] = "onpolicy_paired_positive"
            paired_positive["system2_stop_pair_id"] = pair_id
            sample["system2_stop_pair_id"] = pair_id
            sample["_system2_paired_positive"] = paired_positive
        return sample
