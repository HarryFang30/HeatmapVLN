"""
Tokenized VLN Dataset
=======================

符合 Qwen3-VL 官方实现的数据集。
Tokenization 在 Dataset.__getitem__() 中完成，可以利用 num_workers 并行。

官方流程：
    Dataset.__getitem__(i):
        → 读取图片/视频
        → processor.apply_chat_template()  # tokenization
        → 返回 tokenized 数据
    
    Collator(batch):
        → 只做拼接 (torch.cat)
        → 计算 cumsum_seq_lens
"""

import warnings
# 抑制 Qwen-VL 的 fps 警告（我们使用 nframes 而不是 fps 采样）
warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Qwen3-VL Special Token IDs
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_ID = 151652


def get_rope_index_3(
    spatial_merge_size: int = 2,
    input_ids: torch.LongTensor = None,
    image_grid_thw: torch.LongTensor = None,
    video_grid_thw: torch.LongTensor = None,
    attention_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 Qwen3-VL 的 3D RoPE position IDs（复制自官方实现）"""
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3, input_ids.shape[0], input_ids.shape[1],
            dtype=input_ids.dtype, device=input_ids.device,
        )
        image_index, video_index = 0, 0
        
        for i, input_ids_row in enumerate(total_input_ids):
            input_ids_row = input_ids_row[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_row == VISION_START_ID).squeeze(1)
            vision_tokens = input_ids_row[vision_start_indices + 1]
            image_nums = (vision_tokens == IMAGE_TOKEN_ID).sum()
            video_nums = (vision_tokens == VIDEO_TOKEN_ID).sum()
            input_tokens = input_ids_row.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            
            for _ in range(image_nums + video_nums):
                ed_image = input_tokens.index(IMAGE_TOKEN_ID, st) if IMAGE_TOKEN_ID in input_tokens and remain_images > 0 else len(input_tokens) + 1
                ed_video = input_tokens.index(VIDEO_TOKEN_ID, st) if VIDEO_TOKEN_ID in input_tokens and remain_videos > 0 else len(input_tokens) + 1
                    
                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                    
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            mrope_position_deltas = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas


class TokenizedVLNDataset(Dataset):
    """
    Tokenized VLN Dataset - 符合官方实现
    
    在 __getitem__ 中完成 tokenization，可以利用 DataLoader 的 num_workers 并行。
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        processor,
        spatial_merge_size: int = 2,
    ):
        """
        Args:
            base_dataset: 基础数据集 (VLNTrajectoryDataset 或 VLNSlidingWindowDataset)
            processor: Qwen3-VL processor
            spatial_merge_size: 视觉空间合并大小
        """
        self.base_dataset = base_dataset
        self.processor = processor
        self.spatial_merge_size = spatial_merge_size
        self._call_count = 0  # 用于周期性内存清理
    
    def __len__(self):
        return len(self.base_dataset)
    
    def set_epoch(self, epoch: int):
        """传递 epoch 设置到基础数据集"""
        if hasattr(self.base_dataset, 'set_epoch'):
            self.base_dataset.set_epoch(epoch)
    
    @property
    def hm_size(self):
        return self.base_dataset.hm_size
    
    @hm_size.setter
    def hm_size(self, value):
        self.base_dataset.hm_size = value
    
    def _tensor_to_pil_images(self, tensor: torch.Tensor) -> List[Image.Image]:
        """Convert tensor to PIL images"""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        images = []
        for i in range(tensor.shape[0]):
            frame = tensor[i].permute(1, 2, 0).numpy()
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(frame))
        return images
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        加载并 tokenize 一个样本
        
        关键：tokenization 在这里完成，可以被 DataLoader 并行
        """
        # 1. 从基础数据集获取原始数据
        sample = self.base_dataset[idx]
        
        history_frames = sample['history_frames']  # [K, 3, H, W]
        current_frame = sample['current_frame']    # [3, H, W]
        text = sample.get('text', '')
        
        # 2. 转换为 PIL 图像
        history_pil = self._tensor_to_pil_images(history_frames)
        current_pil = self._tensor_to_pil_images(current_frame)[0]
        
        # 3. 构建 prompt
        if not text:
            text = "Navigate according to the visual observations."
        
        prompt_text = (
            f"You are a navigation assistant. "
            f"The video shows the historical trajectory, and the image shows your current view. "
            f"Instruction: {text}. "
            f"Understand the spatial layout and identify where you came from."
        )
        
        # 4. 构建 messages
        # 使用 nframes 明确指定帧数，避免 fps 采样警告
        content = [
            {"type": "video", "video": history_pil, "nframes": len(history_pil)},
            {"type": "image", "image": current_pil},
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": content}]
        
        # 5. Tokenization（关键步骤！）
        result = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        input_ids = result["input_ids"]  # (1, seq_len)
        seq_len = input_ids.shape[1]
        
        # 6. 计算 position_ids
        image_grid_thw = result.get("image_grid_thw")
        video_grid_thw = result.get("video_grid_thw")
        
        position_ids, _ = get_rope_index_3(
            spatial_merge_size=self.spatial_merge_size,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        
        # 7. 返回 tokenized 数据 + 原始 VLN 数据
        output = {
            # Tokenized LLM inputs
            "input_ids": input_ids,                    # (1, seq_len)
            "position_ids": position_ids,              # (3, 1, seq_len)
            "attention_mask": [seq_len],               # 序列长度（用于 packing）
            "pixel_values": result.get("pixel_values"),
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": result.get("pixel_values_videos"),
            "video_grid_thw": video_grid_thw,
            
            # VLN specific data (用于 downstream heads)
            "history_frames": history_frames,          # [K, 3, H, W] 保留用于可视化/调试
            "current_frame": current_frame,            # [3, H, W]
            "heatmap": sample['heatmap'],              # [Hm, Wm]
            "action": sample['action'],                # [2]
            "action_valid": sample['action_valid'],
            "discrete_action": sample.get('discrete_action', 1),
            "is_stop": sample.get('is_stop', 0.0),
            "is_flipped": sample.get('is_flipped', False),
            "text": text,
            
            # 轨迹数据（如果有）
            "trajectory": sample.get('trajectory'),
            "trajectory_valid": sample.get('trajectory_valid', 0.0),
            "progress": sample.get('progress', 0.0),
        }
        
        # GPU 热力图计算数据（如果有）
        if 'history_poses' in sample:
            output['history_poses'] = sample['history_poses']
            output['current_pose'] = sample['current_pose']
            output['current_depth'] = sample['current_depth']
            output['intrinsics'] = sample['intrinsics']
            output['has_depth'] = sample['has_depth']
            output['has_intrinsics'] = sample['has_intrinsics']
            output['img_size'] = sample['img_size']
        
        # 🔧 内存管理：显式释放中间变量，周期性 malloc_trim
        # tokenization 产生大量临时内存，Python 不会自动归还 OS
        del sample, history_pil, current_pil, messages, content, result
        
        self._call_count += 1
        if self._call_count % 50 == 0:
            import gc
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        
        return output


class FlattenedCollatorForVLN:
    """
    Flattened Collator - 只做拼接，不做 tokenization
    
    符合官方 FlattenedDataCollatorForSupervisedDataset 实现。
    """
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch - 只拼接已经 tokenize 的数据
        """
        batch_size = len(instances)
        
        # 1. 拼接 input_ids
        input_ids = torch.cat([inst["input_ids"] for inst in instances], dim=1)
        
        # 2. 拼接 position_ids
        position_ids = torch.cat([inst["position_ids"] for inst in instances], dim=2)
        
        # 3. 计算 cumsum_seq_lens
        seq_lens = []
        for inst in instances:
            mask = inst["attention_mask"]
            if isinstance(mask, list):
                seq_lens.extend(mask)
            else:
                seq_lens.append(int(mask))
        
        cumsum_seq_lens = torch.tensor([0] + seq_lens, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(cumsum_seq_lens, dim=0, dtype=torch.int32)
        
        # 4. 拼接视觉数据
        images = [inst["pixel_values"] for inst in instances if inst.get("pixel_values") is not None]
        pixel_values = torch.cat(images, dim=0) if images else None
        
        image_grid_thw = [inst["image_grid_thw"] for inst in instances if inst.get("image_grid_thw") is not None]
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if image_grid_thw else None
        
        videos = [inst["pixel_values_videos"] for inst in instances if inst.get("pixel_values_videos") is not None]
        pixel_values_videos = torch.cat(videos, dim=0) if videos else None
        
        video_grid_thw = [inst["video_grid_thw"] for inst in instances if inst.get("video_grid_thw") is not None]
        video_grid_thw = torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
        
        # 5. 处理 history_frames（需要 padding 到相同长度）
        history_frames_list = [inst['history_frames'] for inst in instances]
        max_K = max(hf.shape[0] for hf in history_frames_list)
        
        history_frames_padded = []
        for hf in history_frames_list:
            K, C, H, W = hf.shape
            if K < max_K:
                # Padding: 复制最后一帧
                pad = hf[-1:].expand(max_K - K, C, H, W)
                hf = torch.cat([hf, pad], dim=0)
            history_frames_padded.append(hf)
        
        # 6. 构建 packed batch
        batch = {
            # Packed LLM inputs
            "input_ids": input_ids,                  # (1, total_seq_len)
            "attention_mask": cumsum_seq_lens,       # (num_samples + 1,)
            "position_ids": position_ids,            # (3, 1, total_seq_len)
            "seq_lens": seq_lens,
            "num_samples": batch_size,
            
            # Packed vision data
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            
            # VLN data (batched)
            "history_frames": torch.stack(history_frames_padded, dim=0),  # (B, K, C, H, W) 用于可视化
            "current_frame": torch.stack([inst['current_frame'] for inst in instances], dim=0),
            "heatmap": torch.stack([inst['heatmap'] for inst in instances], dim=0),
            "action": torch.stack([inst['action'] for inst in instances], dim=0),
            "action_valid": torch.tensor([inst['action_valid'] for inst in instances]),
            "discrete_action": torch.tensor([inst.get('discrete_action', 1) for inst in instances], dtype=torch.long),
            "is_stop": torch.tensor([inst.get('is_stop', 0.0) for inst in instances]),
            "is_flipped": torch.tensor([inst.get('is_flipped', False) for inst in instances], dtype=torch.bool),
            "text": [inst['text'] for inst in instances],
        }
        
        # 轨迹数据（如果有）- 健壮性检查：确保所有样本都有轨迹
        trajectories = [inst.get('trajectory') for inst in instances]
        if all(t is not None for t in trajectories):
            batch['trajectory'] = torch.stack(trajectories, dim=0)
            batch['trajectory_valid'] = torch.tensor([inst.get('trajectory_valid', 0.0) for inst in instances])
            batch['progress'] = torch.tensor([inst.get('progress', 0.0) for inst in instances])
        
        # GPU 热力图计算数据（如果有）
        if 'history_poses' in instances[0]:
            batch['history_poses'] = torch.stack([inst['history_poses'] for inst in instances], dim=0)
            batch['current_pose'] = torch.stack([inst['current_pose'] for inst in instances], dim=0)
            batch['current_depth'] = torch.stack([inst['current_depth'] for inst in instances], dim=0)
            batch['intrinsics'] = torch.stack([inst['intrinsics'] for inst in instances], dim=0)
            # has_depth 和 has_intrinsics: 取第一个样本的值（假设 batch 内一致）
            batch['has_depth'] = instances[0].get('has_depth', False)
            batch['has_intrinsics'] = instances[0].get('has_intrinsics', False)
            batch['img_size'] = instances[0].get('img_size', (640, 480))
        
        return batch
