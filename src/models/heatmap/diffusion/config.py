"""
Configuration for Diffusion Heatmap Generation.

This module defines the configuration dataclass for the diffusion-based
heatmap generation head.
"""

from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class DiffusionHeatmapConfig:
    """
    Configuration for DiffusionHeatmapHead.
    
    Defines all hyperparameters for the diffusion-based heatmap generation
    module, including condition encoding, UNet architecture, and diffusion
    scheduler settings.
    
    Attributes:
        # Condition Encoding
        llm_dim: Dimension of LLM token features (Qwen-2.5VL output)
        image_channels: Number of channels in observation image
        cond_dim: Dimension of fused condition vector
        image_size: Size of input observation image (H, W)
        
        # UNet2D Architecture
        in_channels: Input channels for noisy heatmap
        out_channels: Output channels for predicted noise
        block_out_channels: Channel dimensions for each UNet level
        layers_per_block: Number of residual blocks per level
        attention_levels: Which levels to add attention (0-indexed)
        
        # Diffusion Scheduler
        num_train_timesteps: Total training timesteps
        num_inference_steps: Inference steps (can be less than training)
        beta_schedule: Noise schedule type
        prediction_type: What the model predicts ('epsilon' or 'v_prediction')
        
        # Heatmap Output
        heatmap_size: Output heatmap size (H, W)
    """
    
    # ==================== Condition Encoding ====================
    llm_dim: int = 2048                    # Qwen-2.5VL hidden dimension
    image_channels: int = 3                # RGB observation
    cond_dim: int = 512                    # Fused condition dimension
    image_size: Tuple[int, int] = (224, 224)  # Input observation size
    
    # LLM projection
    llm_hidden_dim: int = 1024             # Intermediate projection dim
    llm_pool_method: str = 'attention'     # How to pool LLM sequence ('attention', 'mean', 'first', 'last', 'max')
    llm_pool_num_heads: int = 4            # Number of attention heads for attention pooling
    
    # Image encoder
    image_encoder_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    image_encoder_use_pretrained: bool = False  # Use pretrained ResNet

    # Ablation: disable image encoder to test LLM-only mode
    # Set to False to use only LLM features (no CNN encoding of observation)
    use_image_encoder: bool = True
    
    # ==================== UNet2D Architecture ====================
    in_channels: int = 1                   # Heatmap is single channel
    out_channels: int = 1                  # Predict single channel noise
    
    # Channel dimensions for each encoder/decoder level
    block_out_channels: Tuple[int, ...] = (64, 128, 256)
    
    # Residual blocks per level
    layers_per_block: int = 2
    
    # Attention at deepest levels (indices into block_out_channels)
    attention_levels: Tuple[int, ...] = (2,)  # Attention at 256-channel level
    
    # Normalization
    norm_num_groups: int = 8               # Groups for GroupNorm
    
    # ==================== Diffusion Scheduler ====================
    num_train_timesteps: int = 100         # Training diffusion steps
    num_inference_steps: int = 10          # Inference steps (faster)
    beta_schedule: str = 'squaredcos_cap_v2'  # Cosine schedule
    prediction_type: str = 'epsilon'       # Predict noise
    clip_sample: bool = True               # Clip during sampling
    
    # ==================== Heatmap Output ====================
    heatmap_size: Tuple[int, int] = (64, 64)  # Output heatmap resolution
    
    # ==================== 360° Panorama Support ====================
    # Enable circular padding for equirectangular (360°) panorama images
    # Horizontal: circular padding (left-right boundary is continuous)
    # Vertical: replicate padding (top-bottom is not continuous)
    use_circular_padding: bool = False
    
    # ==================== Training ====================
    # Dropout rate for regularization (used in UNet and condition encoders)
    # Applied to: ImageConditionEncoder, LLMConditionProjector, Fusion MLP, ConditionalUnet2D
    dropout: float = 0.1
    
    # ==================== Classifier-Free Guidance (CFG) ====================
    # CFG 用于增强条件效果，让模型更好地利用条件信息
    # 训练时随机 drop 条件，推理时用无条件预测引导有条件预测
    cfg_drop_prob: float = 0.1       # 训练时随机 drop 条件的概率
    cfg_scale: float = 3.0           # 推理时的引导强度 (1.0 = 无引导)
    
    # ==================== Sequence Cross-Attention Conditioning ====================
    # 双路径条件注入: FiLM (全局向量) + Cross-Attention (序列)
    # 解决 AttentionPooling 将整个 LLM 序列压缩为单个向量的信息瓶颈
    use_sequence_conditioning: bool = False    # 是否启用序列级 cross-attention
    seq_cross_attn_heads: int = 8             # cross-attention head 数
    seq_cross_attn_head_dim: int = 64         # 每个 head 的维度
    
    # ==================== Visibility Head ====================
    # 可见性预测头：判断当前视角是否能看到历史轨迹点
    # 当预测为不可见时，跳过扩散推理直接输出全零热力图
    # 彻底解决假阳性问题（扩散模型从噪声开始，天然倾向于生成非零输出）
    use_visibility_head: bool = False         # 是否启用可见性预测头
    visibility_loss_weight: float = 0.5       # 可见性 BCE loss 的权重
    visibility_threshold: float = 0.5         # 推理时的可见性阈值
    
    # ==================== Spatial Feature Injection ====================
    # 将 CNN encoder 的多尺度空间特征注入 UNet skip connections
    # 解决全局池化导致的空间信息丢失，让 UNet 知道"目标在哪里"
    # 实现方式: CNN 各层特征 -> 1x1 conv 投影 -> bilinear resize -> 加到 skip connection
    use_spatial_injection: bool = False       # 是否启用空间特征注入
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.llm_dim > 0, "llm_dim must be positive"
        assert self.cond_dim > 0, "cond_dim must be positive"
        assert len(self.block_out_channels) >= 2, "Need at least 2 UNet levels"
        assert all(lvl < len(self.block_out_channels) for lvl in self.attention_levels), \
            "attention_levels indices must be valid"
        assert self.num_inference_steps <= self.num_train_timesteps, \
            "inference steps cannot exceed training steps"

