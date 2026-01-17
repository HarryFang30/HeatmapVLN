"""
Qwen3-VL Sequence Packing Module
=================================

基于 Qwen3-VL 官方 fine-tuning 框架实现的 Sequence Packing 功能。
参考：/root/Qwen3-VL/qwen-vl-finetune/

核心技术：
1. Bin Packing: 将多个变长样本打包到固定长度，减少 padding 浪费
2. Flattened Attention Mask: 使用 cumulative sequence lengths 代替 2D mask
3. FlashAttention Varlen: 支持变长序列的高效 attention 计算

与传统 Padding 的对比：
- Padding: [样本A, PAD, PAD, ...], [样本B, PAD, PAD, ...] -> 大量 PAD 浪费显存
- Packing: [样本A, 样本B, 样本C, ...] -> 无 PAD，显存利用率最大化
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Sequence
from dataclasses import dataclass
import logging
import itertools

logger = logging.getLogger(__name__)

# Qwen3-VL Special Token IDs
IMAGE_TOKEN_ID = 151655  # <|image_pad|>
VIDEO_TOKEN_ID = 151656  # <|video_pad|>
VISION_START_ID = 151652  # <|vision_start|>
VISION_END_ID = 151653  # <|vision_end|>
IGNORE_INDEX = -100


@dataclass
class PackedBatch:
    """打包后的批次数据结构"""
    
    # 核心序列数据（拼接后）
    input_ids: torch.Tensor          # (1, total_seq_len)
    attention_mask: torch.Tensor     # cumsum_seq_lens, shape: (num_samples + 1,)
    position_ids: torch.Tensor       # (3, 1, total_seq_len) for M-RoPE
    labels: Optional[torch.Tensor]   # (1, total_seq_len) for SFT
    
    # 视觉数据（拼接后）
    pixel_values: Optional[torch.Tensor]        # 所有图像的 pixel values
    image_grid_thw: Optional[torch.Tensor]      # 所有图像的 grid 信息
    pixel_values_videos: Optional[torch.Tensor] # 所有视频的 pixel values  
    video_grid_thw: Optional[torch.Tensor]      # 所有视频的 grid 信息
    
    # 样本边界信息（用于拆分 hidden states）
    seq_lens: List[int]              # 每个样本的序列长度
    num_samples: int                 # 样本数量


def pad_and_cat_position_ids(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """
    将多个 position_ids 拼接，并 pad 到相同长度
    
    Args:
        tensor_list: List of position_ids, each shape (3, 1, seq_len)
    
    Returns:
        Concatenated position_ids, shape (3, 1, total_seq_len)
    """
    if not tensor_list:
        return None
    
    # 直接在 seq 维度拼接
    return torch.cat(tensor_list, dim=2)


@dataclass
class FlattenedDataCollatorForVLN:
    """
    VLN 任务专用的 Flattened Data Collator
    
    将多个样本拼接成一个长序列，使用 cumulative sequence lengths 作为 attention mask。
    这是 Qwen3-VL 官方 FlattenedDataCollatorForSupervisedDataset 的 VLN 适配版本。
    
    与官方的区别：
    1. 不需要 labels（VLN 不是 SFT 任务）
    2. 需要额外处理 heatmap, action 等 VLN 特定数据
    3. 需要记录样本边界用于拆分 hidden states
    """
    
    tokenizer: Any
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate 多个样本为 packed batch
        
        Args:
            instances: 每个 instance 包含：
                - input_ids: (1, seq_len)
                - position_ids: (3, 1, seq_len)
                - attention_mask: [seq_len] 或 int
                - pixel_values: optional
                - image_grid_thw: optional
                - pixel_values_videos: optional
                - video_grid_thw: optional
        
        Returns:
            packed batch dict
        """
        # 提取核心数据
        input_ids = [instance["input_ids"] for instance in instances]
        position_ids = [instance["position_ids"] for instance in instances]
        attention_masks = [instance["attention_mask"] for instance in instances]
        
        # 计算 cumulative sequence lengths
        # attention_mask 可能是 [seq_len] 或 int(seq_len)
        seq_lens = []
        for mask in attention_masks:
            if isinstance(mask, list):
                seq_lens.extend(mask)
            elif isinstance(mask, int):
                seq_lens.append(mask)
            else:
                seq_lens.append(mask.sum().item() if hasattr(mask, 'sum') else int(mask))
        
        # cumsum_seq_lens: [0, len1, len1+len2, ...]
        cumsum_seq_lens = torch.tensor([0] + seq_lens, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(cumsum_seq_lens, dim=0, dtype=torch.int32)
        
        # 拼接 input_ids
        input_ids = torch.cat(input_ids, dim=1)  # (1, total_seq_len)
        
        # 拼接 position_ids
        position_ids = pad_and_cat_position_ids(position_ids)  # (3, 1, total_seq_len)
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": cumsum_seq_lens,  # cumulative sequence lengths
            "position_ids": position_ids,
            "seq_lens": seq_lens,  # 用于拆分 hidden states
            "num_samples": len(instances),
        }
        
        # 处理图像数据
        images = [
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance and instance["pixel_values"] is not None
        ]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance and instance["image_grid_thw"] is not None
            ]
            if grid_thw:
                batch["image_grid_thw"] = torch.cat(grid_thw, dim=0)
        
        # 处理视频数据
        videos = [
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance and instance["pixel_values_videos"] is not None
        ]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance and instance["video_grid_thw"] is not None
            ]
            if video_grid_thw:
                batch["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)
        
        return batch


def split_packed_hidden_states(
    hidden_states: torch.Tensor,
    seq_lens: List[int],
    pool_method: str = "last",
) -> torch.Tensor:
    """
    将 packed hidden states 拆分为各样本的表示
    
    Args:
        hidden_states: (1, total_seq_len, hidden_dim) packed hidden states
        seq_lens: 每个样本的序列长度
        pool_method: 池化方法
            - "last": 取每个样本的最后一个 token
            - "mean": 平均池化
            - "first": 取每个样本的第一个 token
    
    Returns:
        (num_samples, hidden_dim) 每个样本的表示
    """
    hidden_dim = hidden_states.shape[-1]
    num_samples = len(seq_lens)
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # 计算边界
    cumsum = [0]
    for length in seq_lens:
        cumsum.append(cumsum[-1] + length)
    
    sample_hidden = torch.zeros(num_samples, hidden_dim, device=device, dtype=dtype)
    
    for i, (start, end) in enumerate(zip(cumsum[:-1], cumsum[1:])):
        if end <= start:
            continue
        
        sample_seq = hidden_states[0, start:end, :]  # (seq_len, hidden_dim)
        
        if pool_method == "last":
            sample_hidden[i] = sample_seq[-1]
        elif pool_method == "first":
            sample_hidden[i] = sample_seq[0]
        elif pool_method == "mean":
            sample_hidden[i] = sample_seq.mean(dim=0)
        else:
            raise ValueError(f"Unknown pool method: {pool_method}")
    
    return sample_hidden


def split_packed_vision_hidden_states(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    seq_lens: List[int],
) -> torch.Tensor:
    """
    从 packed hidden states 中提取各样本的视觉 token hidden states
    
    Args:
        hidden_states: (1, total_seq_len, hidden_dim)
        input_ids: (1, total_seq_len)
        seq_lens: 每个样本的序列长度
    
    Returns:
        (num_samples, max_vision_tokens, hidden_dim)
    """
    hidden_dim = hidden_states.shape[-1]
    num_samples = len(seq_lens)
    device = hidden_states.device
    dtype = hidden_states.dtype
    
    # 找到所有视觉 token 的位置
    vision_mask = (input_ids[0] == VIDEO_TOKEN_ID) | (input_ids[0] == IMAGE_TOKEN_ID)
    
    # 计算边界
    cumsum = [0]
    for length in seq_lens:
        cumsum.append(cumsum[-1] + length)
    
    # 统计每个样本的视觉 token 数量
    vision_counts = []
    for start, end in zip(cumsum[:-1], cumsum[1:]):
        count = vision_mask[start:end].sum().item()
        vision_counts.append(count)
    
    max_vision_tokens = max(vision_counts) if vision_counts else 1
    
    # 提取视觉 hidden states
    vision_hidden = torch.zeros(
        num_samples, max_vision_tokens, hidden_dim,
        device=device, dtype=dtype
    )
    
    for i, (start, end) in enumerate(zip(cumsum[:-1], cumsum[1:])):
        sample_vision_mask = vision_mask[start:end]
        vision_indices = sample_vision_mask.nonzero(as_tuple=True)[0]
        n_tokens = len(vision_indices)
        if n_tokens > 0:
            vision_hidden[i, :n_tokens] = hidden_states[0, start + vision_indices, :]
    
    return vision_hidden


def get_rope_index_3(
    spatial_merge_size: int = 2,
    input_ids: torch.LongTensor = None,
    image_grid_thw: torch.LongTensor = None,
    video_grid_thw: torch.LongTensor = None,
    second_per_grid_ts: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Qwen3-VL 的 3D RoPE position IDs
    
    复制自官方实现：/root/Qwen3-VL/qwen-vl-finetune/qwenvl/data/rope2d.py
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    image_token_id = IMAGE_TOKEN_ID
    video_token_id = VIDEO_TOKEN_ID
    vision_start_token_id = VISION_START_ID
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
        
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
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


def replace_attention_with_varlen(model: nn.Module) -> None:
    """
    替换 Qwen3-VL 的 attention forward 函数以支持 varlen FlashAttention
    
    这允许模型处理 packed sequences，使用 cumulative sequence lengths 作为 attention mask。
    
    Args:
        model: Qwen3-VL 模型
    """
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
    except ImportError:
        logger.warning(
            "flash_attn not installed, varlen attention not available. "
            "Install with: pip install flash-attn --no-build-isolation"
        )
        return
    
    import transformers
    from transformers.utils.deprecation import deprecate_kwarg
    from transformers.processing_utils import Unpack
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    from transformers.cache_utils import Cache
    
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
    except ImportError:
        logger.warning("Could not import Qwen3-VL modeling, skipping attention replacement")
        return
    
    def flash_attention_forward_varlen(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        dropout: float = 0.0,
        scaling: float = None,
        sliding_window: int = None,
        softcap: float = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """使用 flash_attn_varlen_func 的 attention forward"""
        
        # FA2 uses non-transposed inputs: batch, head, seq_len, dim -> batch, seq_len, head, dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Squeeze batch dimension for varlen
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)
        cu_seqlens = attention_mask
        
        with torch.no_grad():
            max_seqlen = max(
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ).item()
        
        attn_output = flash_attn_varlen_func(
            query, key, value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        
        attn_output = attn_output.unsqueeze(0)
        return attn_output, None
    
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def qwen3vl_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache = None,
        cache_position: torch.LongTensor = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, None]:
        """Qwen3-VL attention forward with varlen support"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        
        attn_output, attn_weights = flash_attention_forward_varlen(
            self,
            query_states, key_states, value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
    def return_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids, **kwargs):
        """直接返回 attention_mask（cumsum_seq_lens）"""
        return attention_mask
    
    # Apply monkey patches
    try:
        transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention.forward = qwen3vl_attention_forward
        transformers.models.qwen3_vl.modeling_qwen3_vl.create_causal_mask = return_mask
        logger.info("Successfully replaced Qwen3-VL attention with varlen version")
    except Exception as e:
        logger.warning(f"Failed to replace attention: {e}")


class PackedSequenceProcessor:
    """
    Packed Sequence 处理器
    
    封装了 sequence packing 的完整流程，用于 VLN 训练。
    """
    
    def __init__(
        self,
        processor,
        max_seq_length: int = 4096,
        enable_packing: bool = True,
        spatial_merge_size: int = 2,
    ):
        """
        Args:
            processor: Qwen3-VL processor
            max_seq_length: 打包后的最大序列长度
            enable_packing: 是否启用 packing
            spatial_merge_size: 视觉空间合并大小（用于 position IDs 计算）
        """
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_seq_length = max_seq_length
        self.enable_packing = enable_packing
        self.spatial_merge_size = spatial_merge_size
        
        # 设置 left padding 用于批量处理
        self.tokenizer.padding_side = 'left'
    
    def process_single_sample(
        self,
        video_frames: torch.Tensor,
        current_frame: torch.Tensor,
        instruction: str,
    ) -> Dict[str, Any]:
        """
        处理单个样本，返回 tokenized 结果
        
        Args:
            video_frames: (K, C, H, W) 历史视频帧
            current_frame: (C, H, W) 当前帧
            instruction: 导航指令
        
        Returns:
            Dict containing input_ids, position_ids, attention_mask, pixel_values, etc.
        """
        from PIL import Image
        import numpy as np
        
        # Convert tensors to PIL images
        def tensor_to_pil(tensor):
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            images = []
            for i in range(tensor.shape[0]):
                frame = tensor[i].cpu().permute(1, 2, 0).numpy()
                frame = (frame * 255).clip(0, 255).astype(np.uint8)
                images.append(Image.fromarray(frame))
            return images
        
        history_pil = tensor_to_pil(video_frames)
        current_pil = tensor_to_pil(current_frame)[0]
        
        # Build prompt
        if not instruction:
            instruction = "Analyze the spatial relationships in this navigation sequence."
        
        prompt_text = (
            f"You are a navigation assistant. "
            f"The video shows the historical trajectory, and the image shows your current view. "
            f"Instruction: {instruction}. "
            f"Understand the spatial layout and identify where you came from."
        )
        
        # Build messages
        content = [
            {"type": "video", "video": history_pil},
            {"type": "image", "image": current_pil},
            {"type": "text", "text": prompt_text},
        ]
        messages = [{"role": "user", "content": content}]
        
        # Process
        result = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Compute position IDs
        input_ids = result["input_ids"]
        image_grid_thw = result.get("image_grid_thw")
        video_grid_thw = result.get("video_grid_thw")
        
        position_ids, _ = get_rope_index_3(
            spatial_merge_size=self.spatial_merge_size,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        
        result["position_ids"] = position_ids
        result["attention_mask"] = [input_ids.shape[1]]  # seq_len as list
        
        return result
    
    def create_collator(self) -> FlattenedDataCollatorForVLN:
        """创建 Flattened Data Collator"""
        return FlattenedDataCollatorForVLN(tokenizer=self.tokenizer)
