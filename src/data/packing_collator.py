"""
Packing Collator for VLN Training
==================================

基于 Qwen3-VL 官方 fine-tuning 框架实现的 Packing Collator。
将多个样本打包成一个长序列，使用 flash_attn_varlen_func 处理。

核心流程：
1. 接收 batch 的 raw tensors (history_frames, current_frame, ...)
2. 对每个样本调用 processor 进行 tokenization
3. 计算 3D RoPE position IDs
4. 拼接成 packed sequence (1, total_seq_len)
5. 生成 cumsum_seq_lens 作为 attention_mask
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Qwen3-VL Special Token IDs
IMAGE_TOKEN_ID = 151655  # <|image_pad|>
VIDEO_TOKEN_ID = 151656  # <|video_pad|>


def get_rope_index_3(
    spatial_merge_size: int = 2,
    input_ids: torch.LongTensor = None,
    image_grid_thw: torch.LongTensor = None,
    video_grid_thw: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Qwen3-VL 的 3D RoPE position IDs
    复制自官方实现
    """
    image_token_id = IMAGE_TOKEN_ID
    video_token_id = VIDEO_TOKEN_ID
    vision_start_token_id = 151652  # <|vision_start|>
    
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        
        for i, input_ids_row in enumerate(total_input_ids):
            input_ids_row = input_ids_row[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids_row == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids_row[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids_row.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                    
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                    
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


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
            processor: Qwen3-VL processor (AutoProcessor)
            spatial_merge_size: 视觉空间合并大小
            max_seq_length: 最大打包序列长度
        """
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.spatial_merge_size = spatial_merge_size
        self.max_seq_length = max_seq_length
        
        # 设置 left padding
        self.tokenizer.padding_side = 'left'
    
    def _tensor_to_pil_images(self, tensor: torch.Tensor) -> List[Image.Image]:
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
    ) -> Dict[str, Any]:
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
            f"You are a navigation assistant. "
            f"The video shows the historical trajectory, and the image shows your current view. "
            f"Instruction: {instruction}. "
            f"Understand the spatial layout and identify where you came from."
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
    
    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
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
        cumsum_seq_lens = torch.tensor([0] + seq_lens, dtype=torch.int32)
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
            if K < max_K:
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
        
        # 轨迹数据集的额外字段
        if 'trajectory' in batch[0]:
            packed_batch['trajectory'] = torch.stack([s['trajectory'] for s in batch], dim=0)
            packed_batch['trajectory_valid'] = torch.tensor([s.get('trajectory_valid', 0.0) for s in batch])
            packed_batch['progress'] = torch.tensor([s.get('progress', 0.0) for s in batch])
        
        return packed_batch
