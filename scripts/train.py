"""
Training script for VLN Spatial-MLLM Pipeline
Based on BridgeVLA training patterns with VLN-specific enhancements

This script implements the complete training pipeline for first-person inter-frame 
heatmap generation, including space-aware frame sampling and dual-encoder processing.
"""

import os
import sys
import time
import tqdm
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import wandb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.models.spatial_mllm_enhanced import EnhancedSpatialMLLM
from src.data.frame_sampler import SpaceAwareFrameSampler
from src.models.feature_fusion import AdvancedFeatureFusion
from src.models.heatmap.generator import HeatmapGenerator


class VLNTrainer:
    """
    VLN Training Pipeline
    
    Implements multi-stage training:
    1. Pretraining: Focus on heatmap generation with frozen LLM
    2. Fine-tuning: End-to-end spatial reasoning with unfrozen LLM
    """
    
    def __init__(self, config: Dict[str, Any], rank: int = 0, local_rank: int = 0):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.device = f"cuda:{local_rank}"
        
        self.logger = setup_logger(f"trainer_rank{rank}", config['logging']['level'])
        
        # Initialize models
        self._init_models()
        self._init_optimizer()
        self._init_scheduler()
        self._init_loss_functions()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _init_models(self):
        """Initialize all model components"""
        self.logger.info("Initializing VLN models...")
        
        # Frame sampler
        self.frame_sampler = SpaceAwareFrameSampler(
            target_frames=self.config['video']['keyframes'],
            total_frames=self.config['video']['total_frames'],
            **self.config['frame_sampling']
        )
        
        # Enhanced Spatial-MLLM
        self.spatial_mllm = EnhancedSpatialMLLM(
            dinov3_config=self.config['dinov3'],
            vggt_config=self.config['vggt'],
            llm_config=self.config['llm']
        ).to(self.device)
        
        # Feature fusion
        self.feature_fusion = AdvancedFeatureFusion(
            vggt_dim=self.config['vggt']['embed_dim'],
            dinov3_dim=self.config['dinov3']['embed_dim'],
            **self.config['feature_fusion']
        ).to(self.device)
        
        # Heatmap generator
        self.heatmap_generator = HeatmapGenerator(
            **self.config['heatmap']
        ).to(self.device)
        
        # Wrap with DDP if distributed
        if dist.is_initialized() and dist.get_world_size() > 1:
            self.spatial_mllm = DDP(self.spatial_mllm, device_ids=[self.local_rank], 
                                   find_unused_parameters=True)
            self.feature_fusion = DDP(self.feature_fusion, device_ids=[self.local_rank])
            self.heatmap_generator = DDP(self.heatmap_generator, device_ids=[self.local_rank])
        
        self.logger.info("Models initialized successfully")
    
    def _init_optimizer(self):
        """Initialize optimizer with proper parameter grouping"""
        # Collect parameters from all models
        all_params = []
        all_params.extend(list(self.spatial_mllm.parameters()))
        all_params.extend(list(self.feature_fusion.parameters()))
        all_params.extend(list(self.heatmap_generator.parameters()))
        
        # Filter trainable parameters
        trainable_params = [p for p in all_params if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            eps=1e-8
        )
        
        total_params = sum(p.numel() for p in trainable_params)
        self.logger.info(f"Total trainable parameters: {total_params:,}")
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler"""
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['num_epochs']
            )
        elif self.config['training']['scheduler'] == 'linear':
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config['training']['warmup_steps']
            )
        else:
            self.scheduler = None
    
    def _init_loss_functions(self):
        """Initialize loss functions"""
        self.heatmap_loss = nn.MSELoss()
        self.spatial_consistency_loss = nn.L1Loss()
        self.temporal_consistency_loss = nn.L1Loss()
        
        # Loss weights
        self.loss_weights = self.config['loss']
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch"""
        self.spatial_mllm.train()
        self.feature_fusion.train()
        self.heatmap_generator.train()
        
        epoch_losses = defaultdict(list)
        
        pbar = tqdm.tqdm(
            train_loader, 
            disable=(self.rank != 0),
            desc=f"Epoch {epoch}"
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            video_frames = batch['video_frames'].to(self.device)  # [B, N_m, 3, H, W]
            text_instructions = batch['text_instructions']  # List of strings
            target_heatmaps = batch['target_heatmaps'].to(self.device)  # [B, N_views, H, W]
            
            # Forward pass
            losses = self._forward_pass(video_frames, text_instructions, target_heatmaps)
            
            # Backward pass
            total_loss = losses['total_loss']
            total_loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                nn.utils.clip_grad_norm_(
                    [p for model in [self.spatial_mllm, self.feature_fusion, self.heatmap_generator]
                     for p in model.parameters() if p.requires_grad],
                    self.config['training']['gradient_clip']
                )
            
            # Optimizer step
            if (batch_idx + 1) % self.config['training']['accumulation_steps'] == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Log losses
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    epoch_losses[key].append(value.item())
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'heatmap': f"{losses['heatmap_loss'].item():.4f}",
                    'spatial': f"{losses['spatial_loss'].item():.4f}",
                })
            
            # Log to wandb
            if self.rank == 0 and self.global_step % self.config['logging']['log_interval'] == 0:
                wandb.log({
                    **{f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in losses.items()},
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        return {k: sum(v) / len(v) for k, v in epoch_losses.items()}
    
    def _forward_pass(self, video_frames, text_instructions, target_heatmaps):
        """Forward pass through the complete pipeline"""
        batch_size, N_m, C, H, W = video_frames.shape
        
        # Step 1: VGGT processes all frames for geometry extraction
        vggt_features, geometry_info = self.spatial_mllm.extract_geometry_from_all_frames(
            video_frames
        )
        
        # Step 2: Space-aware frame sampling
        selected_indices = []
        for b in range(batch_size):
            indices = self.frame_sampler.sample_keyframes(
                geometry_info={k: v[b] for k, v in geometry_info.items()},
                frame_indices=list(range(N_m))
            )
            selected_indices.append(indices)
        
        # Step 3: Index-select VGGT features and process selected frames with DINOv3
        selected_vggt_features = self._index_select_batch_features(vggt_features, selected_indices)
        selected_frames = self._index_select_batch_frames(video_frames, selected_indices)
        
        dinov3_features = self.spatial_mllm.extract_dinov3_features(selected_frames)
        
        # Step 4: Feature fusion
        fused_features = self.feature_fusion(
            vggt_features=selected_vggt_features,
            dinov3_features=dinov3_features
        )
        
        # Step 5: LLM processing
        llm_outputs = self.spatial_mllm.process_with_llm(
            spatial_features=fused_features,
            text_instructions=text_instructions,
            geometry_context=geometry_info
        )
        
        # Step 6: Generate heatmaps
        predicted_heatmaps = self.heatmap_generator.generate_inter_frame_heatmaps(
            llm_hidden_states=llm_outputs['hidden_states'],
            geometry_info=geometry_info,
            selected_indices=selected_indices
        )
        
        # Calculate losses
        losses = self._calculate_losses(predicted_heatmaps, target_heatmaps, 
                                       selected_indices, geometry_info)
        
        return losses
    
    def _index_select_batch_features(self, features, indices_list):
        """Select features for each batch using different indices"""
        batch_size = features.shape[0]
        selected = []
        for b in range(batch_size):
            selected.append(features[b, indices_list[b]])
        return torch.stack(selected)
    
    def _index_select_batch_frames(self, frames, indices_list):
        """Select frames for each batch using different indices"""
        batch_size = frames.shape[0]
        selected = []
        for b in range(batch_size):
            selected.append(frames[b, indices_list[b]])
        return torch.stack(selected)
    
    def _calculate_losses(self, predicted_heatmaps, target_heatmaps, selected_indices, geometry_info):
        """Calculate all loss components"""
        losses = {}
        
        # Primary heatmap loss
        losses['heatmap_loss'] = self.heatmap_loss(predicted_heatmaps, target_heatmaps)
        
        # Spatial consistency loss (between adjacent frames)
        losses['spatial_loss'] = self._spatial_consistency_loss(predicted_heatmaps)
        
        # Temporal consistency loss
        losses['temporal_loss'] = self._temporal_consistency_loss(predicted_heatmaps)
        
        # Total weighted loss
        losses['total_loss'] = (
            self.loss_weights['heatmap_loss_weight'] * losses['heatmap_loss'] +
            self.loss_weights['spatial_consistency_weight'] * losses['spatial_loss'] +
            self.loss_weights['temporal_consistency_weight'] * losses['temporal_loss']
        )
        
        return losses
    
    def _spatial_consistency_loss(self, heatmaps):
        """Calculate spatial consistency loss between views"""
        if heatmaps.shape[1] < 2:  # Need at least 2 views
            return torch.tensor(0.0, device=self.device)
        
        consistency_loss = 0
        num_pairs = 0
        
        for i in range(heatmaps.shape[1] - 1):
            for j in range(i + 1, heatmaps.shape[1]):
                consistency_loss += self.spatial_consistency_loss(
                    heatmaps[:, i], heatmaps[:, j]
                )
                num_pairs += 1
        
        return consistency_loss / max(num_pairs, 1)
    
    def _temporal_consistency_loss(self, heatmaps):
        """Calculate temporal consistency loss"""
        if heatmaps.shape[1] < 2:
            return torch.tensor(0.0, device=self.device)
        
        temporal_loss = 0
        for i in range(heatmaps.shape[1] - 1):
            temporal_loss += self.temporal_consistency_loss(
                heatmaps[:, i], heatmaps[:, i + 1]
            )
        
        return temporal_loss / max(heatmaps.shape[1] - 1, 1)
    
    def save_checkpoint(self, epoch: int, filepath: str):
        """Save training checkpoint"""
        if self.rank == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': self.global_step,
                'spatial_mllm_state_dict': (
                    self.spatial_mllm.module.state_dict() 
                    if isinstance(self.spatial_mllm, DDP) 
                    else self.spatial_mllm.state_dict()
                ),
                'feature_fusion_state_dict': (
                    self.feature_fusion.module.state_dict()
                    if isinstance(self.feature_fusion, DDP)
                    else self.feature_fusion.state_dict()
                ),
                'heatmap_generator_state_dict': (
                    self.heatmap_generator.module.state_dict()
                    if isinstance(self.heatmap_generator, DDP)
                    else self.heatmap_generator.state_dict()
                ),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'config': self.config
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved: {filepath}")


def train_pipeline(pipeline, config: Dict[str, Any], args, distributed: bool, 
                  rank: int, local_rank: int):
    """
    Main training function called from main.py
    
    Args:
        pipeline: VLN pipeline instance
        config: Configuration dictionary
        args: Command line arguments
        distributed: Whether using distributed training
        rank: Global rank
        local_rank: Local rank
    """
    logger = setup_logger(f"train_rank{rank}", config['logging']['level'])
    logger.info("Starting VLN training pipeline")
    
    # Initialize trainer
    trainer = VLNTrainer(config, rank, local_rank)
    
    # Setup wandb logging
    if rank == 0:
        wandb.init(
            project=config['logging']['wandb_project'],
            entity=config['logging'].get('wandb_entity'),
            config=config,
            name=f"vln_training_{time.strftime('%Y%m%d_%H%M%S')}"
        )
    
    # TODO: Initialize data loaders
    # This needs to be implemented with actual VLN datasets
    logger.warning("Data loaders not yet implemented - using placeholder")
    
    # Training loop
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # TODO: Replace with actual data loader
        # epoch_losses = trainer.train_epoch(train_loader, epoch)
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['logging']['save_dir'],
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            trainer.save_checkpoint(epoch, checkpoint_path)
        
        logger.info(f"Epoch {epoch + 1} completed")
    
    logger.info("Training completed successfully")
    
    if rank == 0:
        wandb.finish()


if __name__ == "__main__":
    # This script can be run standalone for testing
    print("VLN Training Script - Run via main.py for full functionality")