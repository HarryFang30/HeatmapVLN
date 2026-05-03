"""
Packing Collator for VLN Training
==================================

Note: Packing is disabled on the current Qwen2.5-VL training stack.
This module is kept for reference but should use standard batching
(TokenizedVLNDataset + FlattenedCollatorForVLN) instead.
"""

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from .tokenized_dataset import IMAGE_TOKEN_ID, VIDEO_TOKEN_ID, get_rope_index_3  # noqa: F401

logger = logging.getLogger(__name__)


class PackingCollatorForVLN:
    """
    VLN 任务的 Packing Collator

    在 collate 阶段完成 tokenization 和 packing，
    输出可以直接被 forward_packed 使用的数据格式。
    """

    def __init__(
        self,
        processor,
        spatial_merge_size: int = 2,
        max_seq_length: int = 8192,
    ):
        """
        Args:
            processor: Qwen2.5-VL processor (AutoProcessor)
            spatial_merge_size: vision spatial merge size
            max_seq_length: max packed sequence length
        """
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.spatial_merge_size = spatial_merge_size
        self.max_seq_length = max_seq_length

        # 设置 left padding
        self.tokenizer.padding_side = 'left'

    def _tensor_to_pil_images(self, tensor: torch.Tensor) -> list[Image.Image]:
        """
        Convert tensor to list of PIL images

        Args:
            tensor: (K, C, H, W) or (C, H, W)
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        images = []
        for i in range(tensor.shape[0]):
            frame = tensor[i].cpu().permute(1, 2, 0).numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(frame))
        return images

    def _process_single_sample(
        self,
        history_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: str,
    ) -> dict[str, Any]:
        """
        处理单个样本，返回 tokenized 结果
        """
        # 转换为 PIL 图像
        history_pil = self._tensor_to_pil_images(history_frames)
        current_pil = self._tensor_to_pil_images(current_frame)[0]

        # 构建 prompt
        if not instruction:
            instruction = "Navigate according to the visual observations."

        prompt_text = (
            "You are a navigation assistant. "
            "The video shows the historical trajectory from a forward-facing camera. "
            "The image shows your current front view. "
            f"Instruction: {instruction}. "
            "Understand the spatial layout and identify where you came from."
        )

        # 构建 messages
        # 使用 nframes 明确指定帧数，避免 fps 采样警告
        content = [
            {"type": "video", "video": history_pil, "nframes": len(history_pil)},
            {"type": "image", "image": current_pil},
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": content}]

        # 调用 processor
        result = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return result

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        """
        Collate batch，执行 tokenization 和 packing

        Args:
            batch: List of samples, each containing:
                - history_frames: (K, C, H, W)
                - current_frame: (C, H, W)
                - heatmap: (Hm, Wm)
                - action: (2,) or (predict_horizon, 3)
                - text: str
                - ...

        Returns:
            Dict containing packed data for forward_packed
        """
        batch_size = len(batch)

        # ========== 1. Tokenize 每个样本 ==========
        tokenized_samples = []
        for sample in batch:
            result = self._process_single_sample(
                sample['history_frames'],
                sample['current_frame'],
                sample['text'],
            )
            tokenized_samples.append(result)

        # ========== 2. 计算 position_ids 并记录 seq_lens ==========
        processed_samples = []
        for result in tokenized_samples:
            input_ids = result["input_ids"]  # (1, seq_len)
            seq_len = input_ids.shape[1]

            # 获取 grid 信息
            image_grid_thw = result.get("image_grid_thw")
            video_grid_thw = result.get("video_grid_thw")

            # 计算 position_ids
            position_ids, _ = get_rope_index_3(
                spatial_merge_size=self.spatial_merge_size,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

            processed_samples.append({
                "input_ids": input_ids,
                "position_ids": position_ids,  # (3, 1, seq_len)
                "seq_len": seq_len,
                "pixel_values": result.get("pixel_values"),
                "image_grid_thw": image_grid_thw,
                "pixel_values_videos": result.get("pixel_values_videos"),
                "video_grid_thw": video_grid_thw,
            })

        # ========== 3. Pack 成单个序列 ==========
        seq_lens = [s["seq_len"] for s in processed_samples]
        total_seq_len = sum(seq_lens)

        # 检查是否超过最大长度
        if total_seq_len > self.max_seq_length:
            logger.warning(
                f"Total sequence length {total_seq_len} exceeds max {self.max_seq_length}. "
                f"Consider reducing batch_size."
            )

        # 拼接 input_ids: (1, total_seq_len)
        input_ids = torch.cat([s["input_ids"] for s in processed_samples], dim=1)

        # 拼接 position_ids: (3, 1, total_seq_len)
        position_ids = torch.cat([s["position_ids"] for s in processed_samples], dim=2)

        # 计算 cumsum_seq_lens: [0, len1, len1+len2, ...]
        cumsum_seq_lens = torch.tensor([0, *seq_lens], dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(cumsum_seq_lens, dim=0, dtype=torch.int32)

        # ========== 4. 拼接视觉数据 ==========
        # 图像
        images = [s["pixel_values"] for s in processed_samples if s["pixel_values"] is not None]
        if images:
            pixel_values = torch.cat(images, dim=0)
            image_grid_thw = torch.cat(
                [s["image_grid_thw"] for s in processed_samples if s["image_grid_thw"] is not None],
                dim=0
            )
        else:
            pixel_values = None
            image_grid_thw = None

        # 视频
        videos = [s["pixel_values_videos"] for s in processed_samples if s["pixel_values_videos"] is not None]
        if videos:
            pixel_values_videos = torch.cat(videos, dim=0)
            video_grid_thw = torch.cat(
                [s["video_grid_thw"] for s in processed_samples if s["video_grid_thw"] is not None],
                dim=0
            )
        else:
            pixel_values_videos = None
            video_grid_thw = None

        # ========== 5. 处理 VLN 特定数据（非 packed，保持 batch 维度）==========
        # 这些数据用于 downstream heads，不需要 pack
        max_K = max(s['history_frames'].shape[0] for s in batch)
        history_frames_padded = []
        history_mask = []

        for s in batch:
            frames = s['history_frames']
            K = frames.shape[0]
            if max_K > K:
                pad_size = max_K - K
                pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
                frames_padded = torch.cat([frames, pad_frames], dim=0)
                mask = torch.cat([torch.ones(K, dtype=torch.bool), torch.zeros(pad_size, dtype=torch.bool)])
            else:
                frames_padded = frames
                mask = torch.ones(K, dtype=torch.bool)
            history_frames_padded.append(frames_padded)
            history_mask.append(mask)

        # ========== 6. 构建输出 ==========
        packed_batch = {
            # Packed LLM inputs
            "input_ids": input_ids,                    # (1, total_seq_len)
            "attention_mask": cumsum_seq_lens,         # (num_samples + 1,)
            "position_ids": position_ids,              # (3, 1, total_seq_len)
            "seq_lens": seq_lens,                      # List[int]
            "num_samples": batch_size,

            # Packed vision data
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,

            # VLN specific data (batched, not packed)
            "history_frames": torch.stack(history_frames_padded, dim=0),  # (B, K, C, H, W)
            "history_mask": torch.stack(history_mask, dim=0),             # (B, K)
            "current_frame": torch.stack([s['current_frame'] for s in batch], dim=0),
            "heatmap": torch.stack([s['heatmap'] for s in batch], dim=0),
            "action": torch.stack([s['action'] for s in batch], dim=0),
            "action_valid": torch.tensor([s['action_valid'] for s in batch]),
            "discrete_action": torch.tensor([s.get('discrete_action', 1) for s in batch], dtype=torch.long),
            "is_stop": torch.tensor([s.get('is_stop', 0.0) for s in batch]),
            "text": [s['text'] for s in batch],
        }

        if 'current_views' in batch[0]:
            packed_batch['current_views'] = torch.stack(
                [s['current_views'] for s in batch], dim=0)  # [B, 4, C, H, W]
        if 'history_panoramas' in batch[0]:
            packed_batch['history_panoramas'] = torch.stack(
                [s['history_panoramas'] for s in batch], dim=0)  # [B, N, 4, C, H, W]

        # 轨迹数据集的额外字段
        if 'trajectory' in batch[0]:
            packed_batch['trajectory'] = torch.stack([s['trajectory'] for s in batch], dim=0)
            trajectory_valid = [s.get('trajectory_valid', 0.0) for s in batch]
            if torch.is_tensor(trajectory_valid[0]):
                packed_batch['trajectory_valid'] = torch.stack(trajectory_valid, dim=0)
            else:
                packed_batch['trajectory_valid'] = torch.tensor(trajectory_valid)
            packed_batch['progress'] = torch.tensor([s.get('progress', 0.0) for s in batch])

        return packed_batch
