"""
Pydantic v2 configuration schema for HeatmapVLN training configs.

Validates YAML configuration at startup, catching typos and type errors
before they cause cryptic failures deep in the training loop.

Usage::

    from src.config_schema import validate_config, load_and_validate_config

    cfg = validate_config(raw_dict)          # dict -> validated dict
    cfg = load_and_validate_config("x.yaml") # path -> validated dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class _Strict(BaseModel):
    """Base with extra='forbid' to catch misspelled keys."""
    model_config = ConfigDict(extra="forbid")


class _Lenient(BaseModel):
    """Base with extra='allow' for sections where users add ad-hoc keys."""
    model_config = ConfigDict(extra="allow")


# --- Data -------------------------------------------------------------------

class SlidingWindowConfig(_Lenient):
    min_history: int = 5
    num_history_sample: int = 8
    load_depth: bool = True
    cache_poses: bool = True
    sample_stride: int = 1
    clip_level_sampling: bool = True
    samples_per_clip: int = 2
    val_samples_per_clip: int = 2
    defer_heatmap_to_gpu: bool = False
    load_history_frames: bool = True


class TrajectoryConfig(_Lenient):
    min_history: int = 5
    num_history_sample: int = 8
    load_depth: bool = True
    cache_poses: bool = True
    sample_stride: int = 1
    clip_level_sampling: bool = True
    samples_per_clip: int = 8
    val_samples_per_clip: int = 2
    random_subsequence: bool = False
    min_subsequence_length: int = 30
    subsequence_samples_per_clip: int = 3
    predict_horizon: int = 24
    action_scale: float = 4.0
    enable_trajectory_augmentation: bool = True
    load_traj_images: bool = False
    traj_image_size: List[int] = [224, 224]
    use_subinstruction: bool = False
    fgr2r_subinstr_path: Optional[str] = None
    panoramic_vlm_input: bool = True


class DataConfig(_Strict):
    root: str
    train_split: str = "train"
    val_root: Optional[str] = None
    val_split: str = "val"
    image_size: List[int]
    init_hm_size: List[int]
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    dataset_type: str = "sliding_window"
    sliding_window: Optional[SlidingWindowConfig] = None
    trajectory: Optional[TrajectoryConfig] = None
    use_worker_tokenized_collator: Optional[bool] = None

    @field_validator("image_size", "init_hm_size")
    @classmethod
    def _check_size_pair(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError(f"Expected [W, H], got {v}")
        return v

    @field_validator("dataset_type")
    @classmethod
    def _check_dataset_type(cls, v: str) -> str:
        allowed = {"sliding_window", "trajectory"}
        if v not in allowed:
            raise ValueError(f"dataset_type must be one of {allowed}, got '{v}'")
        return v


# --- Model -------------------------------------------------------------------

class LLMConfig(_Lenient):
    model_path: str = "./models/internnav_backbone"
    backbone_type: str = "qwen2_5_vl"
    hidden_dim: int = 3584
    token_dim: int = 896
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    max_video_frames: int = -1
    gradient_checkpointing: bool = False
    enable_packing: bool = False
    max_seq_length: int = 8192
    spatial_merge_size: int = 2
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_num_layers: int = 4
    lora_layer_indices: Optional[List[int]] = None
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None


class HeatmapTrajectoryConfig(_Lenient):
    enable: bool = False
    num_freqs: int = 16
    d_attn: int = 256
    num_heads: int = 4
    num_layers: int = 2
    max_spatial_range: float = 10.0


class HeatmapModelConfig(_Lenient):
    enable: bool = True
    c_vit: int = 1280
    c_llm: int = 3584
    c_fused: int = 256
    vit_layer_indices: List[int] = [7, 15, 23, 31]
    llm_layer_indices: List[int] = [6, 13, 20]
    heatmap_size: List[int] = [64, 64]
    image_size: int = 256
    lambda_vis: float = 1.0
    lambda_coord: float = 0.2
    lambda_kl: float = 0.0
    lambda_peak: float = 1.0
    heatmap_trains_backbone: bool = False
    trajectory: Optional[HeatmapTrajectoryConfig] = None


class NextDiTConfig(_Lenient):
    enabled: bool = False
    vlm_hidden_dim: int = 3584
    latent_emb_size: int = 768
    n_query: int = 4
    dit_dim: int = 384
    dit_layers: int = 12
    dit_heads: int = 6
    dit_kv_heads: int = 6
    dit_ffn_dim_multiplier: float = 2 / 3
    predict_steps: int = 32
    action_dim: int = 3
    num_inference_steps: int = 10
    guidance_scale: float = 1.0
    num_sample_trajs: int = 32
    dav2_ckpt_path: str = ""
    enable_gradient_checkpointing: bool = True
    internnav_system1_path: str = ""
    pretrained_system1_path: Optional[str] = None
    warmup_steps: int = 0


class ActionHeadConfig(_Lenient):
    enable: bool = True
    nextdit: Optional[NextDiTConfig] = None


class ModelConfig(_Lenient):
    type: str = "vln_pipeline"
    device: str = "cuda"
    llm: Optional[LLMConfig] = None
    heatmap: Optional[HeatmapModelConfig] = None
    action_head: Optional[ActionHeadConfig] = None


# --- Optim -------------------------------------------------------------------

class OptimConfig(_Lenient):
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: str = "bf16"
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    batch_size: int = 4
    grad_accum_steps: int = 1

    @field_validator("batch_size")
    @classmethod
    def _positive_batch(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        return v


# --- Loss --------------------------------------------------------------------

class TemperatureScheduleConfig(_Lenient):
    enabled: bool = False
    mode: str = "cosine"
    start: float = 1.0
    end: float = 1.0


class HeatmapVLNLossConfig(_Lenient):
    lambda_vis: float = 1.0
    lambda_coord: float = 0.2
    lambda_kl: float = 0.0
    lambda_peak: float = 1.0
    lambda_neg: float = 0.0
    temperature: float = 1.0
    vis_pos_weight: float = 1.0
    temperature_schedule: Optional[TemperatureScheduleConfig] = None


class LossConfig(_Lenient):
    heatmap_loss_type: str = "heatmap_vln"
    heatmap_vln: Optional[HeatmapVLNLossConfig] = None
    heatmap_weight: float = 1.0
    trajectory_weight: float = 0.0


# --- Training ----------------------------------------------------------------

class TrainingStageConfig(_Lenient):
    name: str
    epochs: int
    hm_size: List[int] = [64, 64]
    heatmap_loss_type: str = "heatmap_vln"
    train_heatmap: Optional[bool] = None
    train_history: Optional[bool] = None
    train_future: Optional[bool] = None
    train_action: bool = True
    trainable_modules: List[str] = []
    frozen_modules: List[str] = []

    @field_validator("epochs")
    @classmethod
    def _positive_epochs(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"epochs must be >= 1, got {v}")
        return v


class TrainingConfig(_Strict):
    stages: List[TrainingStageConfig]


# --- GPU ---------------------------------------------------------------------

class MultiGPUConfig(_Lenient):
    enabled: bool = False
    find_unused_parameters: bool = False


class GPUConfig(_Lenient):
    devices: List[int] = [0]
    backend: str = "nccl"
    multi_gpu: Optional[MultiGPUConfig] = None


# --- Log / Notify ------------------------------------------------------------

class NotifyConfig(_Lenient):
    enabled: bool = False
    platform: str = "feishu"
    webhook_url: str = ""


class LogConfig(_Lenient):
    out_dir: str
    save_every_epochs: int = 1
    vis_every_steps: int = 500
    val_vis_batches: int = 2
    max_ckpts: int = 3
    use_tensorboard: bool = False
    tensorboard_dir: Optional[str] = None
    enable_timing: bool = False
    log_interval: int = 10
    show_gpu_memory: bool = False
    tqdm_ncols: int = 120
    log_level: str = "INFO"
    diag_interval: int = 100
    mid_epoch_save_every: int = 500
    epoch_boundary_cooldown_s: Optional[float] = None
    notify: Optional[NotifyConfig] = None


# --- Validation --------------------------------------------------------------

class ValidationConfig(_Lenient):
    eval_every_epochs: int = 1
    save_best_metric: str = "val_total_loss"
    patience: int = 5
    val_inference_batches: int = 10


# --- Top-level ---------------------------------------------------------------

class TrainConfig(_Lenient):
    """Top-level training configuration schema.

    Uses ``extra='allow'`` at the top level so that ``seed`` and other
    ad-hoc root keys don't cause validation failures, while all nested
    sub-configs use strict or lenient validation as appropriate.
    """
    seed: int = 42
    data: DataConfig
    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig
    gpu: GPUConfig = GPUConfig()
    log: LogConfig
    validation: ValidationConfig = ValidationConfig()


# --- Public API --------------------------------------------------------------

def validate_config(cfg: Dict[str, Any]) -> TrainConfig:
    """Validate a raw config dict against the schema.

    Raises ``pydantic.ValidationError`` with a clear message listing all
    violations if the config is invalid.
    """
    return TrainConfig.model_validate(cfg)


def load_and_validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config file and validate it.

    Returns the original dict (not the Pydantic model) so downstream
    code that expects ``Dict`` keeps working unchanged.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg
