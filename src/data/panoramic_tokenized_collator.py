"""
Panoramic tokenized collator for HeatmapVLN.

This moves Qwen processor/tokenizer work into DataLoader workers so the
training main thread can consume already-tokenized panoramic batches.

When ``n_traj_query > 0`` the collator also:
  1. Passes ``pixel_goal`` (if present in samples) to ``construct_input``.
     In direct mode this includes the coordinate answer; in InternNav mode
     this includes ``↓``, the lookdown user image, and the coordinate answer.
  2. Appends ``n_traj_query`` TRAJ_TOKEN_INDEX placeholder tokens after
     the tokenized sequence, aligned with InternNav's collator flow.
"""

import ctypes
import gc
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

import torch

from src.models.heatmap.input_constructor import (
    construct_input,
    construct_input_stage2,
    find_text_anchor_positions,
    format_structured_pano_assistant_text,
)
from src.models.qwen2_5_vl.integration import TRAJ_TOKEN_INDEX

IGNORE_INDEX = -100
from ._constants import SYSTEM2_ACTION_TEXT as _SYSTEM2_ACTION_TEXT

_libc: ctypes.CDLL | None = None
_TOKENIZER_LOCKS: dict[int, threading.RLock] = {}
_TOKENIZER_LOCKS_GUARD = threading.Lock()


def _get_libc() -> ctypes.CDLL | None:
    """Lazily load libc and cache the handle to avoid repeated dlopen() calls."""
    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL("libc.so.6")
        except OSError:
            pass
    return _libc


def _malloc_trim():
    libc = _get_libc()
    if libc is not None:
        libc.malloc_trim(0)


def _get_tokenizer_lock(tokenizer) -> threading.RLock:
    # Hugging Face fast tokenizers wrap a Rust backend that is not re-entrant.
    # Stage2 adapter prefetch may build batches on several Python threads in
    # the same rank, so collators sharing a tokenizer must serialize tokenizer
    # calls.  Key on the backend object when present so copied wrappers that
    # still share the same backend also share the same lock.
    backend = getattr(tokenizer, "_tokenizer", tokenizer)
    key = id(backend)
    with _TOKENIZER_LOCKS_GUARD:
        lock = _TOKENIZER_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _TOKENIZER_LOCKS[key] = lock
        return lock


def _rss_mb() -> float:
    """Current process RSS in MB (from /proc, zero-overhead)."""
    try:
        with open("/proc/self/statm", "rb") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        return 0.0


class PanoramicTokenizedCollator:
    """Collate panoramic samples and pre-tokenize them for Qwen.

    Args:
        processor: Qwen processor with tokenizer.
        n_traj_query: number of ``<traj>`` placeholder tokens to append
            (aligned with ``nextdit.n_query``).  Set to 0 to disable the
            traj-token mechanism entirely (Stage 1 heatmap-only training).
    """

    def __init__(
        self,
        processor,
        n_traj_query: int = 0,
        *,
        sft_mode: bool = False,
        sft_include_turns: bool = True,
        sft_include_forward: bool = False,
        sft_protocol: str = "direct",
        structured_pano_output: bool = True,
        build_sft_labels: bool = True,
        max_seq_length: int = 8192,
        include_heatmap_targets: bool = True,
        include_history_rel_poses: bool = True,
        retain_raw_panoramic_views: bool = True,
        compute_pano_text_anchor_positions: bool = True,
        heatmap_layout: bool = False,
        force_internnav_prompt: bool = False,
    ):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        # Truncate from left so oldest history tokens are dropped first,
        # preserving the current observation and assistant response.
        self.processor.tokenizer.truncation_side = "left"
        self.n_traj_query = n_traj_query
        self.sft_mode = sft_mode
        self.sft_include_turns = sft_include_turns
        self.sft_include_forward = sft_include_forward
        self.max_seq_length = max_seq_length
        self.sft_protocol = str(sft_protocol).lower()
        if self.sft_protocol not in {"direct", "internnav"}:
            raise ValueError(f"Unsupported System2 SFT protocol: {sft_protocol}")
        self.structured_pano_output = bool(structured_pano_output)
        self.build_sft_labels = bool(build_sft_labels)
        self.include_heatmap_targets = bool(include_heatmap_targets)
        self.include_history_rel_poses = bool(include_history_rel_poses)
        self.retain_raw_panoramic_views = bool(retain_raw_panoramic_views)
        self.compute_pano_text_anchor_positions = bool(compute_pano_text_anchor_positions)
        self.heatmap_layout = bool(heatmap_layout)
        self.force_internnav_prompt = bool(force_internnav_prompt)
        self._call_count = 0
        self._tokenizer_lock = _get_tokenizer_lock(self.processor.tokenizer)

    @staticmethod
    def _stack_optional(batch: list[dict[str, Any]], key: str):
        if key in batch[0]:
            return torch.stack([sample[key] for sample in batch], dim=0)
        return None

    @staticmethod
    def _stack_padded_history_frames(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_k = max(sample["history_frames"].shape[0] for sample in batch)
        heatmap_lengths = []
        for sample in batch:
            for key in ("history_panoramas", "history_rel_poses", "gt_visibility"):
                value = sample.get(key)
                if torch.is_tensor(value) and value.ndim >= 1:
                    heatmap_lengths.append(int(value.shape[0]))
                    break
            else:
                heatmap = sample.get("heatmap")
                if torch.is_tensor(heatmap) and heatmap.ndim >= 4:
                    heatmap_lengths.append(int(heatmap.shape[0]))
                else:
                    heatmap_lengths.append(int(sample["history_frames"].shape[0]))
        max_heatmap_k = max(heatmap_lengths)
        history_frames_padded = []

        for sample in batch:
            frames = sample["history_frames"]
            k = frames.shape[0]
            if k < max_k:
                pad_size = max_k - k
                if k == 0:
                    pad_frames = frames.new_zeros(
                        (pad_size, *frames.shape[1:])
                    )
                else:
                    pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
                frames = torch.cat([frames, pad_frames], dim=0)
            history_frames_padded.append(frames)

        return {
            "history_frames": torch.stack(history_frames_padded, dim=0),
            "history_mask": torch.stack([
                torch.cat([
                    torch.ones(length),
                    torch.zeros(max_heatmap_k - length),
                ])
                for length in heatmap_lengths
            ], dim=0),
        }

    @staticmethod
    def _stack_padded_first_dim(
        batch: list[dict[str, Any]],
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

    @staticmethod
    def _find_last_subsequence(seq: list[int], pattern: list[int]) -> int:
        if not pattern or len(pattern) > len(seq):
            return -1
        for start in range(len(seq) - len(pattern), -1, -1):
            if seq[start:start + len(pattern)] == pattern:
                return start
        return -1

    def _structured_pano_assistant_text(self, sample: dict[str, Any]) -> str | None:
        if not self.structured_pano_output:
            return None
        if sample.get("pano_view_id") is None and sample.get("pano_sample_kind") is None:
            return None
        return format_structured_pano_assistant_text(
            sample.get("pano_view_id"),
            sample.get("pano_pixel_goal"),
            sample_kind=sample.get("pano_sample_kind"),
            is_stop=sample.get("is_stop", 0.0) > 0.5,
        )

    def _assistant_texts_for_sft(self, sample: dict[str, Any]) -> list[str]:
        structured = self._structured_pano_assistant_text(sample)
        if structured is not None:
            return [structured]

        if sample.get("is_stop", 0.0) > 0.5 or int(sample.get("discrete_action", 1)) == 0:
            return ["STOP"]

        pg = sample.get("pixel_goal")
        if pg is not None:
            if self.sft_protocol == "internnav":
                coord_text = f"{int(pg[1])} {int(pg[0])}"
                return ["↓", coord_text]
            coord_text = f"{int(pg[0])} {int(pg[1])}"
            return [coord_text]

        turn_action_text = sample.get("turn_action_text")
        if self.sft_include_turns and isinstance(turn_action_text, str) and turn_action_text:
            return [turn_action_text]
        turn_actions = sample.get("turn_actions")
        if self.sft_include_turns and isinstance(turn_actions, list) and turn_actions:
            return [
                "".join(_SYSTEM2_ACTION_TEXT.get(int(action_code), "") for action_code in turn_actions)
            ]

        discrete_action = int(sample.get("discrete_action", 1))
        if self.sft_include_turns and discrete_action == 2:
            return ["←"]
        if self.sft_include_turns and discrete_action == 3:
            return ["→"]
        if self.sft_include_turns and discrete_action == 5:
            return ["↓"]
        if self.sft_include_forward and discrete_action == 1:
            return ["↑"]
        return []

    def _assistant_sequence_for_labeling(
        self,
        target_text: str,
    ) -> list[int]:
        """Return token ids for the assistant answer content.

        Do not build this from an assistant-only chat template.  Qwen's
        ``apply_chat_template`` injects a default system message for a lone
        assistant turn, so that sequence is not present inside the full
        user+assistant prompt.  Matching the content in the already-tokenized
        row and then adding the following chat end token is more robust.
        """
        tokenizer = self.processor.tokenizer
        return tokenizer.encode(target_text, add_special_tokens=False)

    def _maybe_label_chat_end(
        self,
        labels_row: torch.Tensor,
        input_row: torch.Tensor,
        end: int,
    ) -> None:
        """Include the assistant chat end token in the CE target when present."""
        if end >= input_row.numel():
            return

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if eos_token_id is not None and int(input_row[end].item()) == int(eos_token_id):
            labels_row[end] = input_row[end]

    def _build_sft_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        target_texts: list[list[str]],
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        tokenizer = self.processor.tokenizer
        per_sample_labeled = [False] * len(target_texts)
        per_sample_has_target = [bool(texts) for texts in target_texts]

        for batch_idx, sample_target_texts in enumerate(target_texts):
            if not sample_target_texts:
                continue
            row = input_ids[batch_idx].tolist()
            sample_labeled = False
            for target_text in sample_target_texts:
                match_ids = self._assistant_sequence_for_labeling(target_text)
                if not match_ids:
                    continue

                start = self._find_last_subsequence(row, match_ids)
                if start < 0:
                    # Some tokenizers attach whitespace around short assistant
                    # responses.  Try a tiny set of stable variants before giving up.
                    for variant in (f" {target_text}", f"\n{target_text}"):
                        variant_ids = self._assistant_sequence_for_labeling(variant)
                        start = self._find_last_subsequence(row, variant_ids)
                        if start >= 0:
                            match_ids = variant_ids
                            break
                if start < 0:
                    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
                    start = self._find_last_subsequence(row, target_ids)
                    if start >= 0:
                        match_ids = target_ids
                if start < 0:
                    # Fallback: for multi-line structured output (e.g.
                    # "view: front\\npixel: 128 64"), match each non-empty
                    # line independently and label the full span.  The chat
                    # template may tokenize newlines differently than standalone
                    # encoding, so the full two-line sequence can fail even
                    # though each line is individually tokenisable.
                    lines = [
                        ln.strip()
                        for ln in target_text.split("\n")
                        if ln.strip()
                    ]
                    if len(lines) >= 2:
                        line_starts: list[int] = []
                        line_ids_list: list[list[int]] = []
                        all_matched = True
                        for line in lines:
                            line_ids = tokenizer.encode(line, add_special_tokens=False)
                            ls = self._find_last_subsequence(row, line_ids)
                            if ls < 0:
                                all_matched = False
                                break
                            line_starts.append(ls)
                            line_ids_list.append(line_ids)
                        if all_matched:
                            # Label from the start of the first matched line
                            # through the end of the last matched line.
                            start = line_starts[0]
                            last_ids = line_ids_list[-1]
                            match_ids = row[start:line_starts[-1] + len(last_ids)]
                            # Validate the spanning slice is non-empty.
                            if len(match_ids) == 0:
                                start = -1
                        else:
                            start = -1
                if start < 0:
                    # Last resort: try with add_special_tokens=True in case
                    # the chat template injects BOS/EOS-like tokens around
                    # the assistant content.
                    try:
                        target_ids_special = tokenizer.encode(
                            target_text, add_special_tokens=True,
                        )
                        start = self._find_last_subsequence(row, target_ids_special)
                        if start >= 0:
                            match_ids = target_ids_special
                    except Exception:
                        pass
                if start < 0:
                    continue

                end = start + len(match_ids)
                labels[batch_idx, start:end] = input_ids[batch_idx, start:end]
                self._maybe_label_chat_end(labels[batch_idx], input_ids[batch_idx], end)
                sample_labeled = True

            per_sample_labeled[batch_idx] = sample_labeled

        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_INDEX)

        # Per-sample diagnostics: only warn about samples that *should* have
        # labels (non-empty target_texts) but none were found in the tokenized
        # sequence.  A batch-level RuntimeError is still raised downstream when
        # every sample fails; this warning catches the partial-failure case.
        unlabeled_with_targets = [
            i for i in range(len(target_texts))
            if per_sample_has_target[i] and not per_sample_labeled[i]
        ]
        if unlabeled_with_targets:
            detail_parts: list[str] = []
            for i in unlabeled_with_targets[:3]:
                texts = target_texts[i] if i < len(target_texts) else []
                detail_parts.append(f"idx={i} target={texts}")
            if len(unlabeled_with_targets) > 3:
                detail_parts.append(f"...and {len(unlabeled_with_targets) - 3} more")
            logger.warning(
                "[COLLATOR call=%d] %d/%d samples have NO labels assigned "
                "(target present but not found in tokenized sequence): %s",
                self._call_count,
                len(unlabeled_with_targets),
                len(target_texts),
                "; ".join(detail_parts),
            )

        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self._call_count += 1
        do_log = (self._call_count <= 5) or (self._call_count % 25 == 0)
        pid = os.getpid()

        if do_log:
            rss0 = _rss_mb()

        result = self._stack_padded_history_frames(batch)
        result["current_frame"] = torch.stack([sample["current_frame"] for sample in batch], dim=0)
        if self.include_heatmap_targets:
            result["heatmap"] = self._stack_padded_first_dim(batch, "heatmap")
        result["action"] = torch.stack([sample["action"] for sample in batch], dim=0)
        result["action_valid"] = torch.tensor([sample["action_valid"] for sample in batch])
        result["discrete_action"] = torch.tensor([sample.get("discrete_action", 1) for sample in batch])
        result["is_stop"] = torch.tensor([sample.get("is_stop", 0.0) for sample in batch])
        result["text"] = [sample["text"] for sample in batch]

        for key in ("sample_key", "source_type"):
            if all(key in sample for sample in batch):
                result[key] = [sample[key] for sample in batch]
        for key in (
            "current_pose",
            "current_camera_pose",
            "current_agent_pose",
        ):
            if all(key in sample for sample in batch):
                result[key] = torch.stack(
                    [sample[key] for sample in batch], dim=0
                )
        for key in (
            "history_poses",
            "history_frame_ids",
            "history_age_steps",
        ):
            if all(key in sample for sample in batch):
                result[key] = self._stack_padded_first_dim(batch, key)
        if all("history_valid_mask" in sample for sample in batch):
            result["history_valid_mask"] = self._stack_padded_first_dim(
                batch,
                "history_valid_mask",
            )
            result["history_mask"] = result[
                "history_valid_mask"
            ].float()
        for key in (
            "heatmap_direction_order",
            "history_pose_convention",
        ):
            if all(key in sample for sample in batch):
                result[key] = batch[0][key]

        if "gt_visibility" in batch[0]:
            result["gt_visibility"] = self._stack_padded_first_dim(batch, "gt_visibility")
        if "is_flipped" in batch[0]:
            result["is_flipped"] = torch.tensor([sample.get("is_flipped", False) for sample in batch], dtype=torch.bool)
        if "trajectory" in batch[0]:
            result["trajectory"] = torch.stack([sample["trajectory"] for sample in batch], dim=0)
            trajectory_valid = [sample.get("trajectory_valid", 0.0) for sample in batch]
            if torch.is_tensor(trajectory_valid[0]):
                result["trajectory_valid"] = torch.stack(trajectory_valid, dim=0)
            else:
                result["trajectory_valid"] = torch.tensor(trajectory_valid)
            result["progress"] = torch.tensor([sample.get("progress", 0.0) for sample in batch])
        if self.include_history_rel_poses and "history_rel_poses" in batch[0]:
            result["history_rel_poses"] = self._stack_padded_first_dim(batch, "history_rel_poses")
        if "traj_images" in batch[0]:
            result["traj_images"] = torch.stack([sample["traj_images"] for sample in batch], dim=0)
        if "pixel_goal" in batch[0]:
            result["pixel_goal"] = [sample.get("pixel_goal") for sample in batch]
        if "pano_view_id" in batch[0]:
            result["pano_view_id"] = [sample.get("pano_view_id") for sample in batch]
        if "pano_pixel_goal" in batch[0]:
            result["pano_pixel_goal"] = [sample.get("pano_pixel_goal") for sample in batch]

        if do_log:
            rss1 = _rss_mb()

        has_panoramic = "current_views" in batch[0] and "history_panoramas" in batch[0]
        use_panoramic = has_panoramic and not self.force_internnav_prompt
        use_internnav = "lookdown_frame" in batch[0] and (
            self.force_internnav_prompt or not use_panoramic
        )
        if self.force_internnav_prompt and not use_internnav:
            raise RuntimeError(
                "force_internnav_prompt requires lookdown_frame for every sample"
            )

        if use_panoramic or use_internnav:
            messages_batch = []
            sft_target_texts: list[list[str]] = []
            pano_num_histories = []

            if use_internnav:
                # Stage 2 InternNav-aligned: front-view history + lookdown
                for sample in batch:
                    hf = sample["history_frames"]
                    history_list = [hf[k] for k in range(hf.shape[0])]
                    pg = sample.get("pixel_goal")
                    assistant_texts = self._assistant_texts_for_sft(sample) if self.sft_mode else []
                    assistant_text = assistant_texts[-1] if assistant_texts else None
                    messages_batch.append(
                        construct_input_stage2(
                            history_frames=history_list,
                            current_frame=sample["current_frame"],
                            lookdown_frame=sample["lookdown_frame"],
                            instruction=sample.get("text"),
                            pixel_goal=pg,
                            assistant_text=assistant_text,
                        )
                    )
                    sft_target_texts.append(assistant_texts)
                    pano_num_histories.append(0)
            else:
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
                    pg = sample.get("pano_pixel_goal") or sample.get("pixel_goal")
                    assistant_texts = self._assistant_texts_for_sft(sample) if self.sft_mode else []
                    assistant_text = assistant_texts[-1] if assistant_texts else None
                    lookdown_frame = sample.get("lookdown_frame")
                    use_structured = (
                        self.structured_pano_output
                        and (
                            sample.get("pano_view_id") is not None
                            or sample.get("pano_sample_kind") is not None
                        )
                    )
                    if (
                        self.sft_protocol == "internnav"
                        and pg is not None
                        and lookdown_frame is None
                        and not use_structured
                    ):
                        raise RuntimeError(
                            "InternNav-protocol panoramic sample with pixel_goal "
                            "is missing lookdown_frame."
                        )
                    messages_batch.append(
                        construct_input(
                            current_views=current_views_dict,
                            history_panoramas=history_panoramas_list,
                            instruction=sample.get("text"),
                            pixel_goal=pg,
                            assistant_text=assistant_text,
                            lookdown_frame=lookdown_frame,
                            internnav_protocol=self.sft_protocol == "internnav",
                            structured_pano_output=use_structured,
                            heatmap_layout=self.heatmap_layout,
                        )
                    )
                    if not self.sft_mode and self.sft_protocol == "internnav" and pg is not None and not use_structured:
                        sft_target_texts.append(["↓", f"{int(pg[0])} {int(pg[1])}"])
                    else:
                        sft_target_texts.append(assistant_texts)
                    pano_num_histories.append(len(history_panoramas_list))

                if self.retain_raw_panoramic_views:
                    result["current_views"] = torch.stack(
                        [sample["current_views"] for sample in batch],
                        dim=0,
                    )

            for sample in batch:
                sample.clear()

            if do_log:
                rss2 = _rss_mb()

            has_assistant = any(len(m) > 1 for m in messages_batch)
            with self._tokenizer_lock:
                if self.force_internnav_prompt:
                    # Match the released InternNav Stage-2 path exactly:
                    # render the chat to text first, then invoke the processor
                    # with one flat, sample-major list of independent images.
                    # In particular, history observations are images, not a
                    # Qwen video, so video token/position semantics must never
                    # enter the native System-2 path.
                    rendered_text = self.processor.apply_chat_template(
                        messages_batch,
                        tokenize=False,
                        add_generation_prompt=not has_assistant,
                    )
                    if isinstance(rendered_text, str):
                        rendered_text = [rendered_text]
                    if len(rendered_text) != len(messages_batch):
                        raise RuntimeError(
                            "InternNav chat rendering returned an unexpected "
                            f"batch size: {len(rendered_text)} != "
                            f"{len(messages_batch)}"
                        )

                    native_images = []
                    for messages in messages_batch:
                        for message in messages:
                            content = message.get("content", [])
                            if not isinstance(content, list):
                                continue
                            for item in content:
                                if not isinstance(item, dict):
                                    continue
                                item_type = item.get("type")
                                if item_type == "video":
                                    raise RuntimeError(
                                        "Native InternNav System-2 input must "
                                        "not contain video items"
                                    )
                                if item_type == "image":
                                    native_images.append(item["image"])

                    pano_inputs = self.processor(
                        text=rendered_text,
                        images=native_images,
                        return_tensors="pt",
                        padding=True,
                    )
                    image_grid = pano_inputs.get("image_grid_thw")
                    if (
                        image_grid is None
                        or image_grid.ndim != 2
                        or image_grid.shape[1] != 3
                        or image_grid.shape[0] != len(native_images)
                    ):
                        grid_shape = (
                            None if image_grid is None else tuple(image_grid.shape)
                        )
                        raise RuntimeError(
                            "Native InternNav image_grid_thw does not match "
                            "the independent image list: expected "
                            f"[{len(native_images)},3], got {grid_shape}"
                        )
                    video_grid = pano_inputs.get("video_grid_thw")
                    if video_grid is not None and int(video_grid.shape[0]) > 0:
                        raise RuntimeError(
                            "Native InternNav System-2 processor unexpectedly "
                            "returned video_grid_thw"
                        )
                else:
                    pano_inputs = self.processor.apply_chat_template(
                        messages_batch,
                        tokenize=True,
                        add_generation_prompt=not has_assistant,
                        return_dict=True,
                        return_tensors="pt",
                        padding=True,
                    )
            del messages_batch

            # Only warn when sequence exceeds max length (truncation is disabled).
            seq_len = int(pano_inputs["input_ids"].shape[1])
            max_sl = self.max_seq_length
            if seq_len > max_sl:
                logger.warning(
                    "[COLLATOR call=%d] seq_len=%d > max_seq_len=%d (OVER_LIMIT; NOT TRUNCATED)",
                    self._call_count, seq_len, max_sl,
                )

            if do_log:
                rss3 = _rss_mb()

            if "video_grid_thw" in pano_inputs and pano_inputs["video_grid_thw"] is not None:
                vgt = pano_inputs["video_grid_thw"]
                if vgt.shape[0] > 0 and vgt[:, 0].max() > 1:
                    pano_inputs["video_grid_thw"] = torch.repeat_interleave(vgt, vgt[:, 0], dim=0)
                    pano_inputs["video_grid_thw"][:, 0] = 1

            # Append TRAJ_TOKEN_INDEX placeholders (InternNav-aligned)
            nq = self.n_traj_query
            if nq > 0:
                B, _L = pano_inputs["input_ids"].shape
                traj_tokens = torch.full(
                    (B, nq), TRAJ_TOKEN_INDEX,
                    dtype=pano_inputs["input_ids"].dtype,
                )
                pano_inputs["input_ids"] = torch.cat(
                    [pano_inputs["input_ids"], traj_tokens], dim=1,
                )
                if "attention_mask" in pano_inputs and pano_inputs["attention_mask"] is not None:
                    traj_mask = torch.ones(
                        B, nq, dtype=pano_inputs["attention_mask"].dtype,
                    )
                    pano_inputs["attention_mask"] = torch.cat(
                        [pano_inputs["attention_mask"], traj_mask], dim=1,
                    )

            if self.sft_mode and self.build_sft_labels:
                with self._tokenizer_lock:
                    labels = self._build_sft_labels(
                        pano_inputs["input_ids"],
                        pano_inputs.get("attention_mask"),
                        sft_target_texts,
                    )
                if nq > 0:
                    # Labels were built after appending the TRAJ placeholders,
                    # so this is normally redundant.  Keep it explicit because
                    # TRAJ tokens are latent-query carriers, never LM targets.
                    labels[:, -nq:] = IGNORE_INDEX
                if not torch.any(labels != IGNORE_INDEX):
                    target_summary = [
                        f"sample[{i}]: {texts}"
                        for i, texts in enumerate(sft_target_texts)
                        if texts
                    ][:5]
                    raise RuntimeError(
                        "Panoramic SFT batch has no assistant labels. "
                        "Check pixel_goal/STOP synthesis and tokenizer alignment. "
                        f"Batch target texts: {target_summary}"
                    )
                pano_inputs["labels"] = labels
                result["sft_target_text"] = sft_target_texts

            pano_text_anchor_positions = None
            if not use_internnav and self.compute_pano_text_anchor_positions:
                with self._tokenizer_lock:
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

        if do_log:
            gc.collect(1)
            _malloc_trim()

            rss4 = _rss_mb()
            pano_mb = 0.0
            pi = result.get("pano_inputs")
            if pi is not None:
                for _k, v in pi.items():
                    if isinstance(v, torch.Tensor):
                        pano_mb += v.nelement() * v.element_size() / (1024 * 1024)
            gc_stats = gc.get_stats()
            gen0_collected = gc_stats[0]["collected"] if gc_stats else 0
            gen1_collected = gc_stats[1]["collected"] if len(gc_stats) > 1 else 0
            logger.debug(
                "[COLLATOR pid=%s call=%d] RSS: start=%.0f stack=%.0f PIL→msg=%.0f "
                "tokenize=%.0f gc+trim=%.0f MB | pano_inputs=%.1fMB | "
                "gc_collected: gen0=%d gen1=%d",
                pid, self._call_count, rss0, rss1, rss2,
                rss3, rss4, pano_mb,
                gen0_collected, gen1_collected,
            )

        return result
