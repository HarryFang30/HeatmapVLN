#!/usr/bin/env python3
"""
VLN项目预训练脚本
基于BridgeVLA架构，适配Frame-Indexed Heatmap生成

阶段1：Spatial-Semantic Pretraining
目标：学习3D空间关系和2D语义特征对应关系

作者：VLN团队
参考：BridgeVLA pretrain.py，适配VLN项目需求
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoProcessor
import wandb
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# 添加项目路径
sys.path.append('/home/VLN/Project')
from src.models.spatial_mllm_compat import SpatialMLLMPipeline, SpatialMLLMIntegrationConfig
from src.models.heatmap.generator import HeatmapGenerator
from src.models.heatmap.converter import FrameIndexedHeatmapConverter
from src.data.frame_sampler import SpaceAwareFrameSampler
from src.data.spatial_analysis import SpatialAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def masked_softmax(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    在非零区域进行独立的softmax计算
    参考BridgeVLA实现，适配VLN项目
    """
    mask = (heatmap != 0).float()
    stable_input = heatmap * mask
    exp_vals = torch.exp(stable_input) * mask
    sum_exp = exp_vals.sum(dim=(1, 2), keepdim=True)
    soft_heatmap = exp_vals / (sum_exp + eps)
    return soft_heatmap


def generate_gaussian_heatmap(points: torch.Tensor,
                             image_size: Tuple[int, int] = (224, 224),
                             sigma: float = 5.0,
                             threshold_sigma_times: int = 3) -> torch.Tensor:
    """
    从点坐标生成高斯热力图
    参考BridgeVLA的generate_hm_from_pt，适配VLN需求

    Args:
        points: (num_points, 2) 点坐标
        image_size: (height, width) 图像尺寸
        sigma: 高斯核标准差
        threshold_sigma_times: 阈值倍数

    Returns:
        heatmap: (num_points, height, width) 热力图
    """
    device = points.device
    num_points = points.shape[0]
    height, width = image_size

    # 生成坐标网格
    y_coords = torch.arange(0, height, device=device).float()
    x_coords = torch.arange(0, width, device=device).float()
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # 扩展维度 (1, height, width, 2)
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)
    grid = grid.repeat(num_points, 1, 1, 1)

    # 扩展点坐标 (num_points, 1, 1, 2)
    points_expanded = points.unsqueeze(1).unsqueeze(1)

    # 计算高斯分布
    distances_sq = torch.sum((grid - points_expanded) ** 2, dim=-1)
    heatmap = torch.exp(-distances_sq / (2 * sigma ** 2))

    # 应用阈值
    threshold = np.exp(-0.5 * threshold_sigma_times ** 2)
    heatmap[heatmap < threshold] = 0.0

    # 归一化
    heatmap_sum = torch.sum(heatmap, dim=(1, 2), keepdim=True)
    heatmap = heatmap / (heatmap_sum + 1e-6)

    return heatmap


class VLNPretrainDataset(Dataset):
    """
    VLN预训练数据集
    结合检测数据和空间标注数据，为Frame-Indexed Heatmap学习做准备
    """

    def __init__(self,
                 data_config: Dict,
                 transform=None):
        self.data_config = data_config
        self.transform = transform

        # 初始化数据路径
        self.detection_data_path = data_config.get('detection_data_path', '')
        self.spatial_data_path = data_config.get('spatial_data_path', '')

        # 加载数据索引
        self.data_samples = self._load_data_samples()

        logger.info(f"Loaded {len(self.data_samples)} pretraining samples")

    def _load_data_samples(self) -> List[Dict]:
        """加载数据样本索引"""
        samples = []

        # 加载检测数据
        if os.path.exists(self.detection_data_path):
            with open(self.detection_data_path, 'r') as f:
                detection_data = json.load(f)
            for item in detection_data:
                samples.append({
                    'type': 'detection',
                    'image_path': item['image_path'],
                    'bbox': item['bbox'],  # [x1, y1, x2, y2]
                    'text': item.get('text', ''),
                    'spatial_context': item.get('spatial_context', {})
                })

        # 加载空间标注数据
        if os.path.exists(self.spatial_data_path):
            with open(self.spatial_data_path, 'r') as f:
                spatial_data = json.load(f)
            for item in spatial_data:
                samples.append({
                    'type': 'spatial',
                    'image_path': item['image_path'],
                    'keypoints': item['keypoints'],  # [(x, y), ...]
                    'text': item.get('text', ''),
                    'camera_pose': item.get('camera_pose', {}),
                    'depth_info': item.get('depth_info', {})
                })

        return samples

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 根据数据类型处理标签
        if sample['type'] == 'detection':
            # 边界框转中心点
            bbox = sample['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            # 归一化到[0, 1]
            center_x /= image.size[0] if hasattr(image, 'size') else 224
            center_y /= image.size[1] if hasattr(image, 'size') else 224
            target_points = torch.tensor([[center_x * 224, center_y * 224]], dtype=torch.float32)

        elif sample['type'] == 'spatial':
            # 关键点坐标
            keypoints = sample['keypoints']
            target_points = torch.tensor(keypoints, dtype=torch.float32)
            # 确保坐标在[0, 224]范围内
            target_points = torch.clamp(target_points, 0, 224)

        # 生成目标热力图
        target_heatmap = generate_gaussian_heatmap(
            target_points,
            image_size=(224, 224),
            sigma=5.0
        )

        return {
            'image': image,
            'text': sample['text'],
            'target_heatmap': target_heatmap,
            'data_type': sample['type'],
            'spatial_context': sample.get('spatial_context', {}),
            'target_points': target_points
        }


class VLNPretrainModel(nn.Module):
    """
    VLN预训练模型
    基于Spatial-MLLM，扩展热力图生成能力
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        # 初始化Spatial-MLLM pipeline
        spatial_config = SpatialMLLMIntegrationConfig(
            use_real_llm=True,
            llm_model_path=config['llm_backbone']['model_path'],
            device_allocation=config['gpu_allocation']
        )
        self.spatial_pipeline = SpatialMLLMPipeline(spatial_config)

        # 热力图生成器
        self.heatmap_generator = HeatmapGenerator(
            input_dim=config['model_architecture']['llm_backbone']['hidden_size'],
            output_size=(224, 224),
            sigma=config['model_architecture']['heatmap_generator']['gaussian_sigma']
        )

        # 损失函数
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, batch):
        """
        前向传播

        Args:
            batch: 批次数据
                - images: (B, 3, H, W) 输入图像
                - texts: List[str] 文本指令
                - target_heatmaps: (B, N_points, H, W) 目标热力图

        Returns:
            outputs: 模型输出和损失
        """
        images = batch['image']
        texts = batch['text']
        target_heatmaps = batch['target_heatmap']
        batch_size = images.shape[0]

        # 处理单帧图像（扩展为视频格式以兼容pipeline）
        # (B, 3, H, W) -> (B, 1, 3, H, W)
        video_frames = images.unsqueeze(1)

        # 通过Spatial-MLLM pipeline处理
        pipeline_outputs = self.spatial_pipeline.process_batch(
            video_frames=video_frames,
            instructions=texts,
            return_heatmaps=False,  # 我们自己生成热力图
            return_hidden_states=True
        )

        # 提取LLM隐藏状态
        llm_hidden_states = pipeline_outputs['llm_hidden_states']  # (B, seq_len, hidden_dim)

        # 使用最后一个token的隐藏状态生成热力图
        last_hidden = llm_hidden_states[:, -1, :]  # (B, hidden_dim)

        # 生成预测热力图
        predicted_heatmaps = self.heatmap_generator(last_hidden)  # (B, 1, H, W)

        # 计算损失
        losses = self._compute_losses(predicted_heatmaps, target_heatmaps, batch)

        return {
            'predicted_heatmaps': predicted_heatmaps,
            'target_heatmaps': target_heatmaps,
            'llm_hidden_states': llm_hidden_states,
            'losses': losses,
            'total_loss': losses['total_loss']
        }

    def _compute_losses(self, predicted_heatmaps, target_heatmaps, batch):
        """计算多个损失项"""
        batch_size = predicted_heatmaps.shape[0]

        # 1. 主要的热力图预测损失
        # 将热力图flatten用于交叉熵损失
        pred_flat = predicted_heatmaps.view(batch_size, -1)  # (B, H*W)
        target_flat = target_heatmaps.squeeze(1).view(batch_size, -1)  # (B, H*W)

        # 使用软标签交叉熵
        pred_log_softmax = F.log_softmax(pred_flat, dim=1)
        target_softmax = F.softmax(target_flat, dim=1)
        heatmap_loss = -torch.sum(target_softmax * pred_log_softmax, dim=1).mean()

        # 2. 空间一致性损失
        # 确保预测的热力图具有合理的空间分布
        spatial_consistency_loss = self._compute_spatial_consistency_loss(predicted_heatmaps)

        # 3. 特征对齐损失
        # 确保不同模态的特征在空间上对齐
        alignment_loss = self._compute_feature_alignment_loss(batch)

        # 组合损失
        total_loss = (
            self.config['loss_functions']['pretrain_losses']['gaussian_heatmap_loss']['weight'] * heatmap_loss +
            self.config['loss_functions']['pretrain_losses']['spatial_consistency_loss']['weight'] * spatial_consistency_loss +
            self.config['loss_functions']['pretrain_losses']['cross_frame_alignment_loss']['weight'] * alignment_loss
        )

        return {
            'heatmap_loss': heatmap_loss,
            'spatial_consistency_loss': spatial_consistency_loss,
            'alignment_loss': alignment_loss,
            'total_loss': total_loss
        }

    def _compute_spatial_consistency_loss(self, heatmaps):
        """计算空间一致性损失"""
        # 计算热力图的空间梯度，鼓励平滑性
        grad_x = torch.abs(heatmaps[:, :, :, :-1] - heatmaps[:, :, :, 1:])
        grad_y = torch.abs(heatmaps[:, :, :-1, :] - heatmaps[:, :, 1:, :])

        # 计算总变差损失（Total Variation Loss）
        tv_loss = torch.mean(grad_x) + torch.mean(grad_y)
        return tv_loss

    def _compute_feature_alignment_loss(self, batch):
        """计算特征对齐损失"""
        # 这是一个占位符，可以根据需要实现更复杂的对齐损失
        # 例如，确保3D和2D特征在空间位置上对齐
        return torch.tensor(0.0, device=batch['image'].device)


class VLNPretrainer:
    """VLN预训练器"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = VLNPretrainModel(self.config)

        # 初始化数据集
        self.train_dataset = VLNPretrainDataset(
            self.config['datasets']['pretrain_data']
        )

        # 初始化数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training_stages']['pretrain']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training_stages']['pretrain']['learning_rate'],
            weight_decay=self.config['optimization']['optimizer']['weight_decay']
        )

        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['optimization']['lr_scheduler']['max_steps'],
            eta_min=self.config['optimization']['lr_scheduler']['min_lr']
        )

        # 初始化WandB
        if self.config['monitoring']['wandb']['project']:
            wandb.init(
                project=self.config['monitoring']['wandb']['project'],
                config=self.config,
                tags=self.config['monitoring']['wandb']['tags']
            )

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def train(self):
        """执行预训练"""
        self.model.train()

        num_epochs = self.config['training_stages']['pretrain']['epochs']

        for epoch in range(num_epochs):
            epoch_losses = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for batch_idx, batch in enumerate(pbar):
                # 将数据移到GPU
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # 前向传播
                outputs = self.model(batch)
                loss = outputs['total_loss']

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['optimization']['gradient']['clip_norm']
                )

                self.optimizer.step()
                self.scheduler.step()

                # 记录损失
                epoch_losses.append(loss.item())

                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })

                # 记录到WandB
                if batch_idx % 100 == 0:
                    self._log_metrics(outputs, epoch, batch_idx)

            # 保存检查点
            if (epoch + 1) % self.config['monitoring']['checkpointing']['save_frequency'] == 0:
                self._save_checkpoint(epoch, np.mean(epoch_losses))

            logger.info(f"Epoch {epoch+1} completed. Average loss: {np.mean(epoch_losses):.4f}")

    def _log_metrics(self, outputs, epoch, batch_idx):
        """记录训练指标"""
        metrics = {
            'epoch': epoch,
            'batch': batch_idx,
            'train/total_loss': outputs['losses']['total_loss'].item(),
            'train/heatmap_loss': outputs['losses']['heatmap_loss'].item(),
            'train/spatial_consistency_loss': outputs['losses']['spatial_consistency_loss'].item(),
            'train/alignment_loss': outputs['losses']['alignment_loss'].item(),
            'train/learning_rate': self.scheduler.get_last_lr()[0],
        }

        if wandb.run:
            wandb.log(metrics)

    def _save_checkpoint(self, epoch, avg_loss):
        """保存模型检查点"""
        checkpoint_dir = Path(self.config['monitoring']['checkpointing']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"pretrain_epoch_{epoch+1}.pth"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'avg_loss': avg_loss,
            'config': self.config
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="VLN预训练脚本")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/VLN/Project/configs/training_config.yaml",
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="从检查点恢复训练"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式"
    )

    args = parser.parse_args()

    # 设置调试模式
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        torch.autograd.set_detect_anomaly(True)

    # 初始化预训练器
    trainer = VLNPretrainer(args.config)

    # 如果指定了恢复点，加载检查点
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Resumed from checkpoint: {args.resume_from}")

    # 开始训练
    logger.info("开始VLN预训练...")
    trainer.train()
    logger.info("VLN预训练完成！")


if __name__ == "__main__":
    main()