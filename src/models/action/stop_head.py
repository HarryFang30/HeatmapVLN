"""
Stop Prediction Head for VLN Navigation

This module provides a binary classifier to predict when the agent should stop.
STOP is a discrete action that is hard for diffusion models to predict,
so we use a separate classification head.

Architecture:
    LLM Features (B, seq_len, D) -> Pooling -> MLP -> Sigmoid -> Stop Probability
"""

import logging
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StopPredictionHead(nn.Module):
    """
    Binary classifier for STOP action prediction.
    
    Takes LLM output features and predicts whether the agent should stop.
    Uses focal loss to handle class imbalance (STOP is rare, ~0.6% of actions).
    
    Args:
        input_dim: Input dimension from LLM features
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
        focal_gamma: Gamma parameter for focal loss (higher = more focus on hard examples)
        focal_alpha: Alpha parameter for focal loss (weight for positive class)
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,  # STOP is rare, give it higher weight
    ):
        super().__init__()
        
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Binary output
        )
        
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=0.01)
        nn.init.zeros_(self.classifier[-1].bias)
        
        logger.info(
            f"StopPredictionHead initialized: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, focal_gamma={focal_gamma}, focal_alpha={focal_alpha}"
        )
    
    def forward(
        self,
        llm_features: torch.Tensor,
        gt_stop: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for stop prediction.
        
        Args:
            llm_features: (B, seq_len, D) or (B, D) LLM output features
            gt_stop: Optional (B,) ground truth stop labels (1=STOP, 0=other)
            action_valid: Optional (B,) mask for valid actions
            return_loss: If True and gt_stop provided, return loss dict
            
        Returns:
            If return_loss and gt_stop provided:
                Dict with 'stop_prob', 'stop_logits', 'loss'
            Else:
                (B,) stop probabilities
        """
        # Pool if needed: (B, seq_len, D) -> (B, D)
        if llm_features.dim() == 3:
            pooled = llm_features.mean(dim=1)
        else:
            pooled = llm_features
        
        # Classify: (B, D) -> (B, 1) -> (B,)
        logits = self.classifier(pooled).squeeze(-1)  # (B,)
        stop_prob = torch.sigmoid(logits)
        
        if gt_stop is not None and return_loss:
            loss = self._compute_focal_loss(logits, gt_stop, action_valid)
            return {
                'stop_prob': stop_prob,
                'stop_logits': logits,
                'loss': loss,
            }
        
        return stop_prob
    
    def _compute_focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss for class-imbalanced binary classification.
        
        Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
        
        This helps the model focus on hard-to-classify examples (STOP actions).
        
        Args:
            logits: (B,) raw logits
            targets: (B,) ground truth labels (0 or 1)
            mask: Optional (B,) mask for valid samples
            
        Returns:
            Scalar loss
        """
        device = logits.device
        targets = targets.float().to(device)
        
        # BCE loss per sample
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Compute p_t (probability of correct class)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Alpha weight: alpha for positive, (1-alpha) for negative
        alpha_weight = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        
        # Focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.float().to(device)
            if mask.sum() > 0:
                loss = (focal_loss * mask).sum() / mask.sum()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = focal_loss.mean()
        
        return loss
    
    def predict(self, llm_features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict stop action (binary).
        
        Args:
            llm_features: (B, seq_len, D) or (B, D) LLM features
            threshold: Probability threshold for STOP prediction
            
        Returns:
            (B,) binary predictions (1=STOP, 0=other)
        """
        with torch.no_grad():
            stop_prob = self.forward(llm_features)
            return (stop_prob > threshold).long()


class DiscreteActionHead(nn.Module):
    """
    Optional: Full discrete action classification head.
    
    Predicts one of {STOP, FORWARD, LEFT, RIGHT} actions.
    Can be used as an alternative or complement to the continuous action head.
    
    Args:
        input_dim: Input dimension from LLM features
        hidden_dim: Hidden layer dimension
        num_actions: Number of discrete actions (default 4: STOP, FORWARD, LEFT, RIGHT)
        dropout: Dropout rate
    """
    
    # Action mapping
    ACTION_NAMES = ['STOP', 'FORWARD', 'LEFT', 'RIGHT']
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_actions: int = 4,
        dropout: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.register_buffer('class_weights', class_weights)
        
        # MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        
        logger.info(f"DiscreteActionHead initialized: {num_actions} actions")
    
    def forward(
        self,
        llm_features: torch.Tensor,
        gt_action: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for discrete action prediction.
        
        Args:
            llm_features: (B, seq_len, D) or (B, D) LLM output features
            gt_action: Optional (B,) ground truth action indices
            action_valid: Optional (B,) mask for valid actions
            return_loss: If True and gt_action provided, return loss dict
            
        Returns:
            If return_loss and gt_action provided:
                Dict with 'action_probs', 'action_logits', 'loss', 'predicted_action'
            Else:
                (B, num_actions) action probabilities
        """
        # Pool if needed
        if llm_features.dim() == 3:
            pooled = llm_features.mean(dim=1)
        else:
            pooled = llm_features
        
        # Classify
        logits = self.classifier(pooled)  # (B, num_actions)
        probs = F.softmax(logits, dim=-1)
        predicted = logits.argmax(dim=-1)
        
        if gt_action is not None and return_loss:
            loss = self._compute_loss(logits, gt_action, action_valid)
            return {
                'action_probs': probs,
                'action_logits': logits,
                'predicted_action': predicted,
                'loss': loss,
            }
        
        return probs
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss with optional class weights and masking."""
        device = logits.device
        targets = targets.long().to(device)
        
        # Per-sample loss
        loss_per_sample = F.cross_entropy(
            logits, targets, 
            weight=self.class_weights,
            reduction='none'
        )
        
        # Apply mask
        if mask is not None:
            mask = mask.float().to(device)
            if mask.sum() > 0:
                loss = (loss_per_sample * mask).sum() / mask.sum()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = loss_per_sample.mean()
        
        return loss

