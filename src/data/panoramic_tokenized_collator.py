"""
Panoramic tokenized collator for HeatmapVLN.

This moves Qwen processor/tokenizer work into DataLoader workers so the
training main thread can consume already-tokenized panoramic batches.
"""

import ctypes
import gc
import os
import sys
from typing import Any, Dict, List

import torch

from src.models.heatmap.input_constructor import construct_input, find_text_anchor_positions

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


def _malloc_trim():
    if _libc is not None:
        _libc.malloc_trim(0)


def _rss_mb() -> float:
    """Current process RSS in MB (from /proc, zero-overhead)."""
    try:
        with open("/proc/self/statm", "rb") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


class PanoramicTokenizedCollator:
    """Collate panoramic samples and pre-tokenize them for Qwen."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        self._call_count = 0

    @staticmethod
    def _stack_optional(batch: List[Dict[str, Any]], key: str):
        if key in batch[0]:
            return torch.stack([sample[key] for sample in batch], dim=0)
        return None

    @staticmethod
    def _stack_padded_history_frames(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_k = max(sample["history_frames"].shape[0] for sample in batch)
        history_frames_padded = []
        history_mask = []

        for sample in batch:
            frames = sample["history_frames"]
            k = frames.shape[0]
            if k < max_k:
                pad_size = max_k - k
                pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
                frames = torch.cat([frames, pad_frames], dim=0)
                mask = torch.cat([torch.ones(k), torch.zeros(pad_size)])
            else:
                mask = torch.ones(k)
            history_frames_padded.append(frames)
            history_mask.append(mask)

        return {
            "history_frames": torch.stack(history_frames_padded, dim=0),
            "history_mask": torch.stack(history_mask, dim=0),
        }

    @staticmethod
    def _stack_padded_first_dim(
        batch: List[Dict[str, Any]],
        key: str,
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        max_k = max(sample[key].shape[0] for sample in batch)
        padded_tensors = []

        for sample in batch:
            tensor = sample[key]
            k = tensor.shape[0]
            if k < max_k:
                pad_shape = (max_k - k, *tensor.shape[1:])
                pad_tensor = torch.full(
                    pad_shape,
                    fill_value=pad_value,
                    dtype=tensor.dtype,
                )
                tensor = torch.cat([tensor, pad_tensor], dim=0)
            padded_tensors.append(tensor)

        return torch.stack(padded_tensors, dim=0)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        self._call_count += 1
        do_log = (self._call_count <= 5) or (self._call_count % 25 == 0)
        pid = os.getpid()

        if do_log:
            rss0 = _rss_mb()

        result = self._stack_padded_history_frames(batch)
        result["current_frame"] = torch.stack([sample["current_frame"] for sample in batch], dim=0)
        result["heatmap"] = self._stack_padded_first_dim(batch, "heatmap")
        result["action"] = torch.stack([sample["action"] for sample in batch], dim=0)
        result["action_valid"] = torch.tensor([sample["action_valid"] for sample in batch])
        result["discrete_action"] = torch.tensor([sample.get("discrete_action", 1) for sample in batch])
        result["is_stop"] = torch.tensor([sample.get("is_stop", 0.0) for sample in batch])
        result["text"] = [sample["text"] for sample in batch]

        if "gt_visibility" in batch[0]:
            result["gt_visibility"] = self._stack_padded_first_dim(batch, "gt_visibility")
        if "is_flipped" in batch[0]:
            result["is_flipped"] = torch.tensor([sample.get("is_flipped", False) for sample in batch], dtype=torch.bool)
        if "trajectory" in batch[0]:
            result["trajectory"] = torch.stack([sample["trajectory"] for sample in batch], dim=0)
            result["trajectory_valid"] = torch.tensor([sample.get("trajectory_valid", 0.0) for sample in batch])
            result["progress"] = torch.tensor([sample.get("progress", 0.0) for sample in batch])
        if "history_rel_poses" in batch[0]:
            result["history_rel_poses"] = self._stack_padded_first_dim(batch, "history_rel_poses")

        if do_log:
            rss1 = _rss_mb()

        if "current_views" in batch[0] and "history_panoramas" in batch[0]:
            messages_batch = []
            pano_num_histories = []

            for sample in batch:
                current_views_dict = {
                    name: sample["current_views"][idx]
                    for idx, name in enumerate(("front", "right", "back", "left"))
                }
                history_panoramas_list = [
                    {
                        name: sample["history_panoramas"][hist_idx, view_idx]
                        for view_idx, name in enumerate(("front", "right", "back", "left"))
                    }
                    for hist_idx in range(sample["history_panoramas"].shape[0])
                ]
                messages_batch.append(
                    construct_input(
                        current_views=current_views_dict,
                        history_panoramas=history_panoramas_list,
                        instruction=sample.get("text"),
                    )
                )
                pano_num_histories.append(len(history_panoramas_list))

            result["current_views"] = torch.stack([sample["current_views"] for sample in batch], dim=0)

            for sample in batch:
                sample.clear()

            if do_log:
                rss2 = _rss_mb()

            pano_inputs = self.processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            del messages_batch

            if do_log:
                rss3 = _rss_mb()

            if "video_grid_thw" in pano_inputs and pano_inputs["video_grid_thw"] is not None:
                vgt = pano_inputs["video_grid_thw"]
                if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                    pano_inputs["video_grid_thw"] = torch.repeat_interleave(vgt, vgt[:, 0], dim=0)
                    pano_inputs["video_grid_thw"][:, 0] = 1

            pano_text_anchor_positions = [
                find_text_anchor_positions(
                    pano_inputs["input_ids"][batch_idx:batch_idx + 1],
                    self.processor.tokenizer,
                    num_history=pano_num_histories[batch_idx],
                )
                for batch_idx in range(len(pano_num_histories))
            ]

            result["pano_inputs"] = pano_inputs
            result["pano_num_histories"] = pano_num_histories
            result["pano_text_anchor_positions"] = pano_text_anchor_positions
        else:
            current_views = self._stack_optional(batch, "current_views")
            if current_views is not None:
                result["current_views"] = current_views
            for sample in batch:
                sample.clear()
            if do_log:
                rss2 = rss3 = _rss_mb()

        gc.collect(1)
        _malloc_trim()

        if do_log:
            rss4 = _rss_mb()
            pano_mb = 0.0
            pi = result.get("pano_inputs")
            if pi is not None:
                for k, v in pi.items():
                    if isinstance(v, torch.Tensor):
                        pano_mb += v.nelement() * v.element_size() / (1024 * 1024)
            gc_stats = gc.get_stats()
            gen0_collected = gc_stats[0]["collected"] if gc_stats else 0
            gen1_collected = gc_stats[1]["collected"] if len(gc_stats) > 1 else 0
            print(
                f"[COLLATOR pid={pid} call={self._call_count}] "
                f"RSS: start={rss0:.0f} stack={rss1:.0f} PIL→msg={rss2:.0f} "
                f"tokenize={rss3:.0f} gc+trim={rss4:.0f} MB | "
                f"pano_inputs={pano_mb:.1f}MB | "
                f"gc_collected: gen0={gen0_collected} gen1={gen1_collected}",
                file=sys.stderr,
                flush=True,
            )

        return result
