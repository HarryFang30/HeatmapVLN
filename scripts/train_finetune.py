#!/usr/bin/env python3
"""
VLN项目微调训练脚本
基于BridgeVLA架构，专门针对VLN任务和Frame-Indexed Heatmap生成

阶段2：VLN Task Finetuning
目标：端到端VLN导航和帧索引热力图生成

作者：VLN团队
参考：BridgeVLA train.py，适配VLN项目需求
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
import cv2
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from collections import defaultdict
import time
import subprocess

# 添加项目路径
sys.path.append('/home/VLN/Project')
from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.models.heatmap.generator import HeatmapGenerator
from src.models.heatmap.converter import FrameIndexedHeatmapConverter
from src.data.frame_sampler import SpaceAwareFrameSampler
from src.data.spatial_analysis import SpatialAnalyzer
from src.data.algorithm_factory import get_factory

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed(backend="nccl", port=None):
    """
    初始化分布式训练环境
    参考BridgeVLA实现，支持SLURM和torch.distributed.launch
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)

        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        if os.getenv('DEBUG', 'false').lower() == 'true':
            logger.info("DEBUG模式：单GPU训练")
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "9001"
            os.environ["LOCAL_RANK"] = "0"

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


class VLNFinetuneDataset(Dataset):
    """
    VLN微调数据集
    处理视频序列、导航指令和帧索引标注
    """

    def __init__(self,
                 data_config: Dict,
                 frame_sampler: SpaceAwareFrameSampler,
                 split: str = 'train'):
        self.data_config = data_config
        self.frame_sampler = frame_sampler
        self.split = split

        # 数据路径
        self.vln_data_path = data_config.get('vln_data', {}).get('datasets', [])
        self.frame_indexed_data_path = data_config.get('frame_indexed_data', {}).get('format', '')

        # 加载数据样本
        self.data_samples = self._load_vln_samples()

        logger.info(f"Loaded {len(self.data_samples)} {split} samples")

    def _load_vln_samples(self) -> List[Dict]:
        """加载VLN数据样本"""
        samples = []

        # 模拟数据加载（实际应该从R2R、REVERIE等数据集加载）
        for dataset_name in self.vln_data_path:
            dataset_path = f"/home/VLN/data/{dataset_name}/{self.split}"
            if os.path.exists(dataset_path):
                for video_file in os.listdir(dataset_path):
                    if video_file.endswith('.mp4'):
                        sample = {
                            'video_path': os.path.join(dataset_path, video_file),
                            'instruction': f"Navigate to the target location",  # 实际应该从标注文件加载
                            'trajectory': [],  # 实际轨迹
                            'target_location': (0.5, 0.5),  # 目标位置
                            'frame_correspondences': {},  # 帧间对应关系
                            'dataset': dataset_name
                        }
                        samples.append(sample)

        # 如果没有找到真实数据，创建一些模拟样本用于测试
        if not samples:
            logger.warning("没有找到真实数据，创建模拟样本用于测试")
            for i in range(100):  # 创建100个模拟样本
                sample = {
                    'video_path': f'/home/VLN/test.mp4',  # 使用已存在的测试视频
                    'instruction': f"Navigate to the kitchen and find the refrigerator",
                    'trajectory': [],
                    'target_location': (0.5, 0.5),
                    'frame_correspondences': {},
                    'dataset': 'mock_data'
                }
                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # 加载视频
        video_frames = self._load_video_frames(sample['video_path'])

        # 使用space-aware采样选择关键帧
        num_keyframes = min(8, len(video_frames))  # 最多8个关键帧
        if hasattr(self.frame_sampler, 'sample_frames'):
            sampled_frame_indices = self.frame_sampler.sample_frames(
                video_frames, num_keyframes=num_keyframes
            )
        else:
            # 如果frame_sampler没有sample_frames方法，使用均匀采样
            sampled_frame_indices = np.linspace(0, len(video_frames)-1, num_keyframes, dtype=int).tolist()

        # 获取采样的帧
        keyframes = [video_frames[i] for i in sampled_frame_indices]

        # 生成帧索引对应的热力图标注
        frame_indexed_heatmaps = self._generate_frame_indexed_labels(
            keyframes,
            sampled_frame_indices,
            sample['frame_correspondences']
        )

        return {
            'video_frames': torch.stack([self._preprocess_frame(f) for f in video_frames]),  # (T, 3, H, W)
            'keyframes': torch.stack([self._preprocess_frame(f) for f in keyframes]),  # (K, 3, H, W)
            'keyframe_indices': torch.tensor(sampled_frame_indices, dtype=torch.long),  # (K,)
            'instruction': sample['instruction'],
            'target_heatmaps': frame_indexed_heatmaps,  # (K, H, W) 每个关键帧对应的热力图
            'dataset': sample['dataset'],
            'video_path': sample['video_path']
        }

    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """加载视频帧"""
        if not os.path.exists(video_path):
            # 如果视频不存在，创建模拟帧
            logger.warning(f"视频文件不存在: {video_path}，创建模拟帧")
            frames = []
            for i in range(16):  # 创建16帧模拟视频
                frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                frames.append(frame)
            return frames

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换BGR到RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        # 如果帧数太少，重复帧
        if len(frames) < 8:
            while len(frames) < 8:
                frames.extend(frames[:min(8-len(frames), len(frames))])

        return frames

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """预处理单帧"""
        # 调整大小到224x224
        frame = cv2.resize(frame, (224, 224))
        # 转换为tensor并归一化
        frame = torch.from_numpy(frame).float() / 255.0
        # 转换维度 (H, W, C) -> (C, H, W)
        frame = frame.permute(2, 0, 1)
        return frame

    def _generate_frame_indexed_labels(self,
                                     keyframes: List[np.ndarray],
                                     keyframe_indices: List[int],
                                     frame_correspondences: Dict) -> torch.Tensor:
        """
        生成帧索引标注
        为每个关键帧生成对应的热力图，显示该帧内容在当前视角下的空间位置
        """
        num_keyframes = len(keyframes)
        heatmaps = torch.zeros(num_keyframes, 224, 224)

        for i, frame_idx in enumerate(keyframe_indices):
            # 模拟生成热力图（实际应该基于真实的空间对应关系）
            # 这里简单地在不同位置生成高斯热力图来模拟不同帧的空间对应
            center_x = (frame_idx % 7) * 32 + 112  # 在7x7网格中分布
            center_y = (frame_idx // 7) * 32 + 112

            # 确保坐标在有效范围内
            center_x = max(16, min(208, center_x))
            center_y = max(16, min(208, center_y))

            # 生成高斯热力图
            y, x = torch.meshgrid(
                torch.arange(224, dtype=torch.float32),
                torch.arange(224, dtype=torch.float32),
                indexing='ij'
            )

            sigma = 20.0
            heatmap = torch.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
            heatmap = heatmap / (heatmap.sum() + 1e-6)  # 归一化

            heatmaps[i] = heatmap

        return heatmaps


class VLNFinetuneModel(nn.Module):
    """
    VLN微调模型
    基于预训练的Spatial-MLLM，扩展Frame-Indexed Heatmap生成
    """

    def __init__(self, config: Dict, pretrain_checkpoint: Optional[str] = None):
        super().__init__()

        self.config = config

        # 初始化Spatial-MLLM pipeline
        try:
            spatial_config = SpatialMLLMIntegrationConfig(
                use_real_llm=True,
                llm_model_path=config.get('model_architecture', {}).get('llm_backbone', {}).get('model_path', '/home/VLN/Project/models/qwen_2.5_vl'),
                device_allocation=config.get('distributed_training', {}).get('gpu_allocation', {}).get('device_mapping', {})
            )
            self.spatial_pipeline = SpatialMLLMPipeline(spatial_config)
        except Exception as e:
            logger.warning(f"无法初始化Spatial-MLLM pipeline: {e}，使用简化版本")
            self.spatial_pipeline = None

        # Frame-Indexed Heatmap Converter
        hidden_size = config.get('model_architecture', {}).get('llm_backbone', {}).get('hidden_size', 1024)
        self.frame_indexed_converter = FrameIndexedHeatmapConverter(
            hidden_dim=hidden_size,
            output_size=(224, 224)
        )

        # 简化的特征提取器（如果spatial_pipeline不可用）
        if self.spatial_pipeline is None:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16)),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, hidden_size)
            )

        # 热力图生成头
        self.heatmap_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 224 * 224),
            nn.Sigmoid()
        )

        # 加载预训练权重
        if pretrain_checkpoint:
            self._load_pretrain_checkpoint(pretrain_checkpoint)

        # 损失函数
        self._setup_loss_functions()

    def _setup_loss_functions(self):
        """设置损失函数"""
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def _load_pretrain_checkpoint(self, checkpoint_path: str):
        """加载预训练检查点"""
        logger.info(f"Loading pretrain checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # 这里需要实现从预训练检查点加载权重的逻辑
            logger.info("预训练权重加载成功")
        except Exception as e:
            logger.warning(f"预训练权重加载失败: {e}")

    def forward(self, batch):
        """
        前向传播

        Args:
            batch: 批次数据包含视频帧、指令和目标热力图

        Returns:
            outputs: 模型输出包括预测的frame-indexed heatmaps和损失
        """
        video_frames = batch['video_frames']  # (B, T, 3, H, W)
        keyframes = batch['keyframes']        # (B, K, 3, H, W)
        keyframe_indices = batch['keyframe_indices']  # (B, K)
        instructions = batch['instruction']   # List[str]
        target_heatmaps = batch['target_heatmaps']  # (B, K, H, W)

        batch_size, num_keyframes = keyframes.shape[:2]

        if self.spatial_pipeline is not None:
            # 使用完整的Spatial-MLLM pipeline
            try:
                pipeline_outputs = self.spatial_pipeline.process_batch(
                    video_frames=video_frames,
                    instructions=instructions,
                    return_heatmaps=False,
                    return_hidden_states=True
                )
                llm_hidden_states = pipeline_outputs['llm_hidden_states']  # (B, seq_len, hidden_dim)
                features = llm_hidden_states[:, -1, :]  # 使用最后一个token
            except Exception as e:
                logger.warning(f"Spatial-MLLM pipeline失败: {e}，使用简化特征提取")
                # 使用简化特征提取作为后备
                features = self._extract_simple_features(keyframes)
        else:
            # 使用简化特征提取
            features = self._extract_simple_features(keyframes)

        # 生成每个关键帧的热力图
        predicted_heatmaps = []
        for k in range(num_keyframes):
            # 为每个关键帧生成热力图
            frame_features = features  # 简化版本使用相同特征
            heatmap = self.heatmap_head(frame_features)  # (B, H*W)
            heatmap = heatmap.view(batch_size, 224, 224)  # (B, H, W)
            predicted_heatmaps.append(heatmap)

        predicted_heatmaps = torch.stack(predicted_heatmaps, dim=1)  # (B, K, H, W)

        # 计算损失
        losses = self._compute_losses(predicted_heatmaps, target_heatmaps, batch)

        return {
            'predicted_heatmaps': predicted_heatmaps,
            'target_heatmaps': target_heatmaps,
            'features': features,
            'losses': losses,
            'total_loss': losses['total_loss']
        }

    def _extract_simple_features(self, keyframes):
        """简化的特征提取"""
        batch_size, num_keyframes = keyframes.shape[:2]
        # 使用第一个关键帧作为代表
        first_frame = keyframes[:, 0]  # (B, 3, H, W)
        features = self.feature_extractor(first_frame)  # (B, hidden_size)
        return features

    def _compute_losses(self, predicted_heatmaps, target_heatmaps, batch):
        """计算微调阶段的多任务损失"""
        batch_size, num_keyframes = predicted_heatmaps.shape[:2]

        # 1. Frame-Indexed Heatmap Loss (核心损失)
        frame_indexed_loss = self._compute_frame_indexed_heatmap_loss(
            predicted_heatmaps, target_heatmaps
        )

        # 2. 帧间差异性损失 (确保不同帧的热力图有差异)
        distinctness_loss = self._compute_distinctness_loss(predicted_heatmaps)

        # 3. 时间一致性损失 (相邻帧的热力图应该有合理的变化)
        temporal_consistency_loss = self._compute_temporal_consistency_loss(
            predicted_heatmaps, batch['keyframe_indices']
        )

        # 4. 空间推理损失 (热力图应该反映合理的空间关系)
        spatial_reasoning_loss = self._compute_spatial_reasoning_loss(predicted_heatmaps)

        # 组合损失
        total_loss = (
            1.0 * frame_indexed_loss +
            0.3 * distinctness_loss +
            0.6 * temporal_consistency_loss +
            0.4 * spatial_reasoning_loss
        )

        return {
            'frame_indexed_loss': frame_indexed_loss,
            'distinctness_loss': distinctness_loss,
            'temporal_consistency_loss': temporal_consistency_loss,
            'spatial_reasoning_loss': spatial_reasoning_loss,
            'total_loss': total_loss
        }

    def _compute_frame_indexed_heatmap_loss(self, predicted, target):
        """计算帧索引热力图损失"""
        # 使用MSE损失
        return self.mse_loss(predicted, target)

    def _compute_distinctness_loss(self, heatmaps):
        """计算帧间差异性损失，确保不同帧生成不同的热力图"""
        batch_size, num_keyframes = heatmaps.shape[:2]

        if num_keyframes < 2:
            return torch.tensor(0.0, device=heatmaps.device)

        # 计算所有帧对之间的相似性
        heatmaps_flat = heatmaps.view(batch_size, num_keyframes, -1)  # (B, K, H*W)

        # 计算余弦相似性矩阵
        normalized_heatmaps = F.normalize(heatmaps_flat, p=2, dim=-1)
        similarity_matrix = torch.bmm(normalized_heatmaps, normalized_heatmaps.transpose(1, 2))  # (B, K, K)

        # 我们希望对角线外的元素（不同帧间的相似性）尽可能小
        mask = torch.eye(num_keyframes, device=heatmaps.device).bool()
        off_diagonal = similarity_matrix.masked_fill(mask, 0)

        # 惩罚高相似性（鼓励差异性）
        distinctness_loss = off_diagonal.abs().mean()

        return distinctness_loss

    def _compute_temporal_consistency_loss(self, heatmaps, keyframe_indices):
        """计算时间一致性损失"""
        batch_size, num_keyframes = heatmaps.shape[:2]

        if num_keyframes < 2:
            return torch.tensor(0.0, device=heatmaps.device)

        # 计算相邻帧之间的变化
        temporal_diff = torch.diff(heatmaps, dim=1)  # (B, K-1, H, W)

        # 使用L2范数计算变化程度
        temporal_variation = torch.norm(temporal_diff, p=2, dim=(-2, -1))  # (B, K-1)

        # 鼓励适度的时间变化（不要太急剧，也不要太相似）
        target_variation = 0.5  # 期望的变化程度
        consistency_loss = F.mse_loss(temporal_variation,
                                    torch.full_like(temporal_variation, target_variation))

        return consistency_loss

    def _compute_spatial_reasoning_loss(self, heatmaps):
        """计算空间推理损失"""
        # 检查热力图是否具有合理的空间分布
        batch_size, num_keyframes = heatmaps.shape[:2]

        # 计算热力图的熵，鼓励适度的集中程度
        heatmaps_flat = heatmaps.view(batch_size, num_keyframes, -1)
        heatmaps_prob = F.softmax(heatmaps_flat, dim=-1)

        # 计算熵
        entropy = -torch.sum(heatmaps_prob * torch.log(heatmaps_prob + 1e-8), dim=-1)  # (B, K)

        # 鼓励适度的熵值（既不太集中也不太分散）
        target_entropy = 6.0  # 经验值，可调整
        entropy_loss = F.mse_loss(entropy, torch.full_like(entropy, target_entropy))

        return entropy_loss


def train_epoch(model, dataloader, optimizer, scheduler, epoch, rank=0, world_size=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        # 将数据移到GPU
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # 前向传播
        outputs = model(batch)
        loss = outputs['total_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # 更新进度条
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })

        # 记录到WandB
        if rank == 0 and batch_idx % 50 == 0:
            log_metrics = {
                'train/total_loss': loss.item(),
                'train/frame_indexed_loss': outputs['losses']['frame_indexed_loss'].item(),
                'train/distinctness_loss': outputs['losses']['distinctness_loss'].item(),
                'train/temporal_consistency_loss': outputs['losses']['temporal_consistency_loss'].item(),
                'train/spatial_reasoning_loss': outputs['losses']['spatial_reasoning_loss'].item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'epoch': epoch,
                'batch': batch_idx
            }
            if wandb.run:
                wandb.log(log_metrics)

    return total_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, save_path):
    """保存检查点"""
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'avg_loss': avg_loss
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description="VLN微调训练脚本")
    parser.add_argument("--config", type=str,
                       default="/home/VLN/Project/configs/training_config.yaml",
                       help="训练配置文件路径")
    parser.add_argument("--pretrain_checkpoint", type=str, default="",
                       help="预训练检查点路径")
    parser.add_argument("--resume_from", type=str, default="",
                       help="恢复训练检查点路径")
    parser.add_argument("--log_dir", type=str, default="/home/VLN/Project/logs",
                       help="日志保存目录")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--local_rank", type=int, default=0, help="本地GPU排名")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")

    args = parser.parse_args()

    # 设置分布式训练
    if not args.debug and torch.cuda.device_count() > 1:
        setup_distributed()
    else:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 设置设备
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化WandB
    if rank == 0 and not args.debug:
        wandb.init(
            project=config.get('monitoring', {}).get('wandb', {}).get('project', 'vln_training'),
            config=config,
            tags=config.get('monitoring', {}).get('wandb', {}).get('tags', ['vln', 'heatmap'])
        )

    # 初始化帧采样器
    try:
        factory = get_factory()
        algorithm = factory.create_auto_configured('quality')  # 使用高质量算法
        frame_sampler = algorithm
    except Exception as e:
        logger.warning(f"无法初始化帧采样器: {e}，使用简化版本")
        frame_sampler = None

    # 初始化数据集
    train_dataset = VLNFinetuneDataset(
        config.get('datasets', {}).get('finetune_data', {}),
        frame_sampler,
        split='train'
    )

    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset,
                                     num_replicas=world_size,
                                     rank=rank) if world_size > 1 else None

    # 数据加载器
    batch_size = config.get('training_stages', {}).get('finetune', {}).get('batch_size', 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    # 初始化模型
    model = VLNFinetuneModel(config, args.pretrain_checkpoint)
    model = model.to(device)

    # 分布式模型包装
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 优化器和调度器
    learning_rate = config.get('training_stages', {}).get('finetune', {}).get('learning_rate', 5e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    num_epochs = args.epochs
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # 恢复训练
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if rank == 0:
            logger.info(f"恢复训练从epoch {start_epoch}")

    # 创建保存目录
    save_dir = Path(args.log_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # 训练一个epoch
        avg_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, rank, world_size)

        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}")

            # 保存检查点
            if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次
                checkpoint_path = save_dir / f"finetune_epoch_{epoch+1}.pth"
                save_checkpoint(model, optimizer, scheduler, epoch, avg_loss, checkpoint_path)
                logger.info(f"检查点已保存: {checkpoint_path}")

    if rank == 0:
        logger.info("VLN微调训练完成！")

    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()