"""
Multi-Stage Training Script for VLN Heatmap Model

Implements training pipeline as specified in run_training.md:
- Stage A: Warmup (freeze backbone, train head+renderer)
- Stage B/C: Finetune (unfreeze with LoRA, increase resolution)

Key features:
- Resolution curriculum (64→128→224)
- Freeze/unfreeze control
- Parameter grouping (head_lr vs lora_lr)
- Visualization and checkpointing
"""

import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
import random
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import logging

from src.data.vln_heatmap_adapter import VLNHeatmapDataset
from src.models.vln_heatmap_model import VLNHeatmapModel
from src.utils.losses import kl_ce_loss, mse_heatmap_loss, focal_kl_ce_loss, heatmap_ce_from_logits

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def variable_length_collate_fn(batch):
    """
    Custom collate function to handle variable-length frame sequences and heatmaps.
    Pads all sequences to the maximum length in the batch.
    """
    # Find max frame count and max heatmap count in this batch
    max_T = max(sample['frames'].shape[0] for sample in batch)
    max_K = max(sample['gt_heatmaps'].shape[0] for sample in batch)

    # Prepare batched tensors
    batch_frames = []
    batch_text = []
    batch_gt_heatmaps = []
    batch_mask = []
    batch_meta = []
    batch_frame_mask = []  # mask indicating valid frames

    for sample in batch:
        frames = sample['frames']  # [T, 3, H, W]
        T = frames.shape[0]

        # Pad frames to max_T if needed
        if T < max_T:
            pad_size = max_T - T
            pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)  # Repeat last frame
            frames_padded = torch.cat([frames, pad_frames], dim=0)
            frame_mask = torch.cat([torch.ones(T), torch.zeros(pad_size)])
        else:
            frames_padded = frames
            frame_mask = torch.ones(T)

        # Pad heatmaps to max_K if needed
        gt_heatmaps = sample['gt_heatmaps']  # [K, Hm, Wm]
        mask = sample['mask']  # [K]
        K = gt_heatmaps.shape[0]

        if K < max_K:
            pad_size = max_K - K
            Hm, Wm = gt_heatmaps.shape[1], gt_heatmaps.shape[2]
            pad_heatmaps = torch.zeros(pad_size, Hm, Wm)
            pad_mask = torch.zeros(pad_size)
            gt_heatmaps_padded = torch.cat([gt_heatmaps, pad_heatmaps], dim=0)
            mask_padded = torch.cat([mask, pad_mask], dim=0)
        else:
            gt_heatmaps_padded = gt_heatmaps
            mask_padded = mask

        batch_frames.append(frames_padded)
        batch_text.append(sample['text'])
        batch_gt_heatmaps.append(gt_heatmaps_padded)
        batch_mask.append(mask_padded)
        batch_meta.append(sample['meta'])
        batch_frame_mask.append(frame_mask)

    return {
        'frames': torch.stack(batch_frames, dim=0),  # [B, max_T, 3, H, W]
        'text': batch_text,                          # List of strings
        'gt_heatmaps': torch.stack(batch_gt_heatmaps, dim=0),  # [B, max_K, Hm, Wm]
        'mask': torch.stack(batch_mask, dim=0),      # [B, max_K]
        'meta': batch_meta,                          # List of dicts
        'frame_mask': torch.stack(batch_frame_mask, dim=0)  # [B, max_T]
    }


def create_dataloaders(config: Dict, hm_size: Tuple[int, int]) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and val dataloaders with specified heatmap size.

    Args:
        config: Configuration dictionary
        hm_size: Heatmap size (H, W) for this stage

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_cfg = config['data']

    # Create datasets
    train_dataset = VLNHeatmapDataset(
        root=data_cfg['root'],
        split='train',
        frames_per_clip=data_cfg['frames_per_clip'],
        heatmap_per_clip=data_cfg['heatmap_per_clip'],
        image_size=tuple(data_cfg['image_size']),
        hm_size=hm_size
    )

    val_dataset = VLNHeatmapDataset(
        root=data_cfg['root'],
        split='val',
        frames_per_clip=data_cfg['frames_per_clip'],
        heatmap_per_clip=data_cfg['heatmap_per_clip'],
        image_size=tuple(data_cfg['image_size']),
        hm_size=hm_size
    )

    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['optim']['batch_size'],
        shuffle=True,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        drop_last=True,
        collate_fn=variable_length_collate_fn  # Handle variable-length sequences
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['optim']['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
        drop_last=False,
        collate_fn=variable_length_collate_fn  # Handle variable-length sequences
    )

    logger.info(f"Dataloaders created: train={len(train_dataset)}, val={len(val_dataset)}, hm_size={hm_size}")

    return train_loader, val_loader


def build_param_groups(model: nn.Module, config: Dict, stage_cfg: Dict) -> list:
    """
    Build parameter groups with different learning rates.

    Args:
        model: VLNHeatmapModel
        config: Configuration dictionary
        stage_cfg: Current stage configuration (for use_gru and freeze_llm)

    Returns:
        List of parameter groups for optimizer
    """
    optim_cfg = config['optim']
    head_lr = optim_cfg['head_lr']
    lora_lr = optim_cfg['lora_lr']
    wd = optim_cfg['weight_decay']

    # First, freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    param_groups = []
    use_gru = stage_cfg.get('use_gru', False)
    freeze_backbone = stage_cfg.get('freeze_llm', True)

    # 1) Head and renderer parameters (always trainable)
    head_params = []
    renderer_params = []
    for n, p in model.named_parameters():
        if n.startswith('head'):
            p.requires_grad = True
            head_params.append(p)
        elif 'renderer' in n:
            p.requires_grad = True
            renderer_params.append(p)

    param_groups.append({
        'params': head_params + renderer_params,
        'lr': head_lr,
        'weight_decay': 1e-4,  # Lower WD for head/renderer
        'name': 'head_renderer'
    })

    # 2) Temporal aggregation (GRU + fuse_proj, if use_gru=True)
    temporal_params = []
    if use_gru:
        for n, p in model.named_parameters():
            if n.startswith('temporal') or n.startswith('fuse_proj'):
                p.requires_grad = True
                temporal_params.append(p)
        if temporal_params:
            param_groups.append({
                'params': temporal_params,
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'name': 'temporal_fuse'
            })

    # 3) Encoder parameters (only if not frozen; partial unfreeze for layer4+proj)
    encoder_params = []
    if not freeze_backbone:
        for n, p in model.named_parameters():
            # Only unfreeze encoder.layer4 and encoder.proj (last ResNet block + projection)
            if 'encoder.backbone' in n and 'layer4' in n:
                p.requires_grad = True
                encoder_params.append(p)
            elif n.startswith('encoder.proj'):
                p.requires_grad = True
                encoder_params.append(p)
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': lora_lr,
                'weight_decay': wd,
                'name': 'encoder_layer4_proj'
            })

    # Log parameter counts
    total_params = sum(sum(p.numel() for p in group['params']) for group in param_groups)
    logger.info(f"Parameter groups created: {len(param_groups)} groups, {total_params:,} total params")
    for group in param_groups:
        n_params = sum(p.numel() for p in group['params'])
        logger.info(f"  {group['name']}: {n_params:,} params, lr={group['lr']:.2e}")

    return param_groups


def build_scheduler(optimizer, config: Dict, steps_per_epoch: int, stage_epochs: int):
    """Build learning rate scheduler"""
    optim_cfg = config['optim']
    scheduler_type = optim_cfg.get('scheduler', 'cosine')
    warmup_ratio = optim_cfg.get('warmup_ratio', 0.05)

    total_steps = steps_per_epoch * stage_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Warmup + Cosine
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        scheduler = None

    logger.info(f"Scheduler created: {scheduler_type}, warmup_steps={warmup_steps}")

    return scheduler


def create_loss_fn(config: Dict, use_logits: bool = True):
    """
    Create loss function based on config.

    Args:
        config: Configuration dictionary
        use_logits: If True, use logits-space loss (RECOMMENDED for training)
                   If False, use probability-space loss (for backward compatibility)
    """
    loss_cfg = config['loss']
    loss_type = loss_cfg['type']

    if use_logits:
        # RECOMMENDED: Logits-space loss for better gradient flow
        if loss_type == 'kl_ce':
            loss_fn = heatmap_ce_from_logits
            logger.info("Loss function: Logits-space CE (RECOMMENDED)")
        elif loss_type == 'mse':
            # For MSE, we still need probabilities
            use_logits = False
            loss_fn = mse_heatmap_loss
            logger.info("Loss function: MSE (using probabilities)")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    else:
        # Probability-space loss (for backward compatibility)
        if loss_type == 'kl_ce':
            if loss_cfg.get('focal', {}).get('enabled', False):
                alpha = loss_cfg['focal']['alpha']
                gamma = loss_cfg['focal']['gamma']
                def loss_fn(pred, target, mask=None):
                    return focal_kl_ce_loss(pred, target, mask, alpha=alpha, gamma=gamma)
                logger.info(f"Loss function: Focal KL-CE (alpha={alpha}, gamma={gamma})")
            else:
                loss_fn = kl_ce_loss
                logger.info("Loss function: KL-CE (probability space)")
        elif loss_type == 'mse':
            loss_fn = mse_heatmap_loss
            logger.info("Loss function: MSE")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_fn, use_logits


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, config, scaler=None, use_logits=True):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        frames = batch['frames'].to(device)        # [B, T, 3, H, W]
        targets = batch['gt_heatmaps'].to(device)  # [B, K, Hm, Wm]
        mask = batch['mask'].to(device)            # [B, K]

        # Forward pass with mixed precision
        use_amp = config['optim']['amp'] in ['bf16', 'fp16']
        dtype = torch.bfloat16 if config['optim']['amp'] == 'bf16' else torch.float16

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type, dtype=dtype):
                preds, _ = model(frames, return_logits=use_logits)
                loss = loss_fn(preds, targets, mask=mask)

            # Backward with gradient scaling
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Gradient clipping
            if config['optim']['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optim']['grad_clip'])

            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            preds, _ = model(frames, return_logits=use_logits)
            loss = loss_fn(preds, targets, mask=mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            if config['optim']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['optim']['grad_clip'])

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Log progress
        if batch_idx % config['log']['log_interval'] == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}, LR: {lr:.2e}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, val_loader, loss_fn, device, use_logits=True):
    """Validate model"""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            frames = batch['frames'].to(device)
            targets = batch['gt_heatmaps'].to(device)
            mask = batch['mask'].to(device)

            preds, _ = model(frames, return_logits=use_logits)
            loss = loss_fn(preds, targets, mask=mask)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, epoch, stage_name, config, val_loss):
    """Save training checkpoint"""
    out_dir = Path(config['log']['out_dir']) / 'checkpoints'
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'stage': stage_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'config': config
    }

    ckpt_path = out_dir / f"checkpoint_{stage_name}_epoch_{epoch}.pt"
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")

    # Keep only last N checkpoints
    max_ckpts = config['log']['max_ckpts']
    checkpoints = sorted(out_dir.glob(f"checkpoint_{stage_name}_*.pt"))
    if len(checkpoints) > max_ckpts:
        for old_ckpt in checkpoints[:-max_ckpts]:
            old_ckpt.unlink()
            logger.info(f"Removed old checkpoint: {old_ckpt}")


def main(args):
    """Main training function"""
    # Load config
    config = load_config(args.config)
    set_seed(config['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create loss function (use logits-space loss for better gradient flow)
    loss_fn, use_logits = create_loss_fn(config, use_logits=True)

    # Initialize model once (will update hm_size per stage)
    init_hm_size = tuple(config['data']['init_hm_size'])
    model = VLNHeatmapModel(
        k_heatmaps=config['data']['heatmap_per_clip'],
        hm_size=init_hm_size,
        vision_dim=config['model']['vision_dim'],
        agg=config['model']['agg'],
        use_lora=False,  # Will enable per stage
        models_root=config['model']['models_root']
    ).to(device)

    logger.info(f"Model initialized: {model.get_trainable_parameters()['total']:,} parameters")

    # Load initial checkpoint if provided
    if args.init_ckpt and Path(args.init_ckpt).exists():
        ckpt = torch.load(args.init_ckpt, map_location='cpu')
        # Try different keys for checkpoint structure
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        logger.info(f"Loaded init checkpoint: {args.init_ckpt}")
    elif args.init_ckpt:
        logger.warning(f"Init checkpoint not found: {args.init_ckpt}")

    # Gradient scaler for mixed precision
    use_amp = config['optim']['amp'] in ['bf16', 'fp16']
    scaler = GradScaler('cuda') if use_amp else None

    # Loop through training stages
    for stage in config['training']['stages']:
        stage_name = stage['name']
        logger.info("=" * 80)
        logger.info(f"Starting stage: {stage_name}")
        logger.info("=" * 80)

        # Update heatmap size (curriculum)
        hm_size = tuple(stage['hm_size'])
        if hm_size != model.hm_size:
            logger.info(f"Updating heatmap size: {model.hm_size} → {hm_size}")
            model.update_hm_size(hm_size)

        # Freeze/unfreeze backbone
        freeze_backbone = stage['freeze_llm']
        model.freeze_backbone(freeze=freeze_backbone)
        logger.info(f"Backbone {'frozen' if freeze_backbone else 'unfrozen'}")

        # Create dataloaders for this stage
        train_loader, val_loader = create_dataloaders(config, hm_size)

        # Build parameter groups and optimizer (pass stage config)
        param_groups = build_param_groups(model, config, stage)
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=config['optim']['weight_decay']
        )

        # Build scheduler
        scheduler = build_scheduler(
            optimizer, config,
            steps_per_epoch=len(train_loader),
            stage_epochs=stage['epochs']
        )

        # Train for stage epochs
        for epoch in range(stage['epochs']):
            logger.info(f"\nStage {stage_name} - Epoch {epoch + 1}/{stage['epochs']}")

            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                loss_fn, device, config, scaler, use_logits=use_logits
            )

            # Validate
            val_loss = validate(model, val_loader, loss_fn, device, use_logits=use_logits)

            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save checkpoint
            if (epoch + 1) % config['log']['save_every_epochs'] == 0:
                save_checkpoint(model, optimizer, scheduler, epoch + 1, stage_name, config, val_loss)

        logger.info(f"Stage {stage_name} completed\n")

    logger.info("=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage VLN heatmap training")
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to config file')
    parser.add_argument('--init-ckpt', type=str, default=None,
                       help='Initialize model weights from this checkpoint (e.g., Stage-A best)')

    args = parser.parse_args()
    main(args)