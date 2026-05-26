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
_SYSTEM2_ACTION_TEXT = {
    0: "STOP",
    1: "↑",
    2: "←",
    3: "→",
    5: "↓",
}

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
    ):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        self.n_traj_query = n_traj_query
        self.sft_mode = sft_mode
        self.sft_include_turns = sft_include_turns
        self.sft_include_forward = sft_include_forward
        self.sft_protocol = str(sft_protocol).lower()
        if self.sft_protocol not in {"direct", "internnav"}:
            raise ValueError(f"Unsupported System2 SFT protocol: {sft_protocol}")
        self.structured_pano_output = bool(structured_pano_output)
        self._call_count = 0

    @staticmethod
    def _stack_optional(batch: list[dict[str, Any]], key: str):
        if key in batch[0]:
            return torch.stack([sample[key] for sample in batch], dim=0)
        return None

    @staticmethod
    def _stack_padded_history_frames(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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
            coord_text = f"{int(pg[0])} {int(pg[1])}"
            if self.sft_protocol == "internnav":
                return ["↓", coord_text]
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

        for batch_idx, sample_target_texts in enumerate(target_texts):
            if not sample_target_texts:
                continue
            row = input_ids[batch_idx].tolist()
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
                    if start < 0:
                        continue
                    match_ids = target_ids

                end = start + len(match_ids)
                labels[batch_idx, start:end] = input_ids[batch_idx, start:end]
                self._maybe_label_chat_end(labels[batch_idx], input_ids[batch_idx], end)

        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, IGNORE_INDEX)
        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
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
            trajectory_valid = [sample.get("trajectory_valid", 0.0) for sample in batch]
            if torch.is_tensor(trajectory_valid[0]):
                result["trajectory_valid"] = torch.stack(trajectory_valid, dim=0)
            else:
                result["trajectory_valid"] = torch.tensor(trajectory_valid)
            result["progress"] = torch.tensor([sample.get("progress", 0.0) for sample in batch])
        if "history_rel_poses" in batch[0]:
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

        use_panoramic = "current_views" in batch[0] and "history_panoramas" in batch[0]
        use_internnav = "lookdown_frame" in batch[0] and not use_panoramic

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
                        )
                    )
                    if not self.sft_mode and self.sft_protocol == "internnav" and pg is not None and not use_structured:
                        sft_target_texts.append(["↓", f"{int(pg[0])} {int(pg[1])}"])
                    else:
                        sft_target_texts.append(assistant_texts)
                    pano_num_histories.append(len(history_panoramas_list))

                result["current_views"] = torch.stack([sample["current_views"] for sample in batch], dim=0)

            for sample in batch:
                sample.clear()

            if do_log:
                rss2 = _rss_mb()

            has_assistant = any(len(m) > 1 for m in messages_batch)
            pano_inputs = self.processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=not has_assistant,
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

            if self.sft_mode:
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
                    raise RuntimeError(
                        "Panoramic SFT batch has no assistant labels. "
                        "Check pixel_goal/STOP synthesis and tokenizer alignment."
                    )
                pano_inputs["labels"] = labels
                result["sft_target_text"] = sft_target_texts

            pano_text_anchor_positions = None
            if not use_internnav:
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
