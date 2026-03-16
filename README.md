# HeatmapVLN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900.svg)

**Heatmap and Trajectory Prediction for Vision-Language Navigation based on Qwen3.5-9B**

[Quick Start](#quick-start) •
[Architecture](#model-architecture) •
[Training](#training) •
[Configuration](#configuration) •
[FAQ](#faq)

</div>

---

## Overview

HeatmapVLN is a deep learning framework for Vision-Language Navigation (VLN) tasks. Given 360° panoramic observations (front/right/back/left), a history frame sequence, and a natural language instruction, the model predicts projection heatmaps of each history camera position onto the current view, along with continuous trajectory prediction and task progress estimation.

The core architecture uses a **Coarse-to-Fine** two-stage heatmap generation pipeline: a frozen Qwen3.5-9B backbone extracts multi-layer features, which are fused via DPT-Lite modules to first produce an 8x8 coarse localization, then refined into a 64x64 fine-grained heatmap. Only ~2M parameters need to be trained.

<p align="center">
  <img src="assets/architecture.png" width="800" alt="Architecture">
</p>

### Key Features

| Feature | Description |
|:--------|:------------|
| **Coarse-to-Fine Heatmap** | 8x8 coarse localization -> 64x64 fine heatmap via cosine similarity + ConvTranspose refinement |
| **Multi-Layer Feature Fusion** | DPT-Lite fusion of multi-layer ViT features (16x16) and LLM features (8x8) |
| **360° Panoramic Support** | Simultaneous heatmap and visibility prediction across 4 directions (front/right/back/left) |
| **Visibility Prediction** | MLP visibility head determines whether a history frame is visible in the current view, gating the heatmap output |
| **Trajectory Prediction** | 24-step continuous trajectory prediction (x, y, theta) via Transformer Decoder + DDPM |
| **Progress Estimation** | Task completion progress regression (0-1) via 3-layer MLP |
| **Qwen3.5-9B Backbone** | Frozen VLM backbone with hook-based intermediate ViT/LLM feature extraction |
| **LoRA Fine-Tuning (Optional)** | LoRA adapters on the last N layers of Qwen3.5 for enhanced spatial reasoning |
| **FGR2R Sub-Instructions** | Dynamic sub-instruction matching for random subsequence sampling |
| **Trajectory Augmentation** | Random rotation/scaling of trajectories during training for better generalization |
| **Modular Design** | Heatmap/trajectory/progress heads can be independently enabled or disabled |

---

## Table of Contents

- [Quick Start](#quick-start)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Model Preparation](#model-preparation)
- [Usage Guide](#usage-guide)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
- [Model Architecture](#model-architecture)
  - [Overall Architecture](#overall-architecture)
  - [Heatmap Generation Module](#heatmap-generation-module)
  - [Trajectory Prediction Module](#trajectory-prediction-module)
- [Dataset](#dataset)
  - [Data Format](#data-format)
  - [Sampling Strategies](#sampling-strategies)
- [Configuration](#configuration)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Quick Start

### Requirements

- Python 3.11+
- PyTorch 2.7+
- CUDA 12.8+
- 40GB+ GPU memory (A100/H100 recommended)

### Installation

**Option 1: Docker Deployment (Recommended)**

```bash
./docker/docker-run.sh
```

See the [Docker Guide](docker/DOCKER.md) for details.

**Option 2: Local Installation**

```bash
# Create conda environment
conda create -n heatmapvln python=3.11 -y
conda activate heatmapvln

# Install PyTorch (CUDA 12.8)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Optional: Install FlashAttention 2 (recommended)
pip install flash-attn --no-build-isolation
```

### Model Preparation

Download Qwen3.5 model weights:

```bash
# From HuggingFace
huggingface-cli download Qwen/Qwen3.5-VL-9B --local-dir models/qwen_3.5

# Or from ModelScope
modelscope download Qwen/Qwen3.5-VL-9B --local_dir models/qwen_3.5
```

### Quick Validation

```bash
# Verify installation
python scripts/train.py --config configs/train_config.yaml --dry-run

# Quick training test
python scripts/train.py --config configs/train_config.yaml --epochs 2 --max-batches 5
```

---

## Usage Guide

### Training

```bash
# Full training (heatmap + trajectory + progress)
python scripts/train.py --config configs/train_config.yaml

# Heatmap-only training (lightweight)
python scripts/train.py --config configs/train_heatmap_config.yaml

# Resume from checkpoint
python scripts/train.py --config configs/train_config.yaml --auto-resume
```

**Common Arguments:**

| Argument | Description | Example |
|:---------|:------------|:--------|
| `--config` | Config file path | `configs/train_config.yaml` |
| `--resume` | Resume from a specific checkpoint | `--resume ckpts/e005.pth` |
| `--auto-resume` | Auto-resume from the latest checkpoint | |
| `--dry-run` | Build model only, skip training | |
| `--epochs` | Number of training epochs | `--epochs 10` |

**Background Training:**

```bash
# Using tmux (recommended)
tmux new -s train
python scripts/train.py --config configs/train_config.yaml
# Ctrl+B D to detach, tmux attach -t train to reattach

# Using nohup
nohup python -u scripts/train.py --config configs/train_config.yaml > train.log 2>&1 &
```

**TensorBoard Monitoring:**

```bash
tensorboard --logdir /root/tf-logs --port=6006
```

<details>
<summary>Key TensorBoard Metrics</summary>

| Category | Metric | Description |
|:---------|:-------|:------------|
| **Heatmap Loss** | `train/heatmap_loss` | Total heatmap loss |
| | `train/vis_loss` | Visibility BCE loss |
| | `train/peak_loss` | Softmax CE localization loss |
| | `train/coord_loss` | Coordinate auxiliary loss |
| | `train/neg_loss` | Invisible-view suppression loss |
| **Trajectory Loss** | `train/trajectory_loss` | Trajectory diffusion loss |
| **Progress Loss** | `train/progress_loss` | Progress regression loss |
| **Heatmap Diagnostics** | `diag/pred_heatmap_max` | Predicted heatmap max value |
| | `diag/pred_heatmap_mean` | Predicted heatmap mean value |
| **Trajectory Diagnostics** | `diag/trajectory_ade` | Average Displacement Error |
| | `diag/trajectory_fde` | Final Displacement Error |
| **Progress Diagnostics** | `diag/progress_mae` | Progress MAE |
| | `diag/progress_pred_mean` | Predicted progress mean |
| | `diag/progress_gt_mean` | Ground truth progress mean |
| **Resource Monitoring** | `diag/gpu_memory_gb` | GPU memory usage |

</details>

### Inference

```bash
# Inference on a dataset clip
python scripts/inference.py \
  --clip /path/to/clip \
  --config configs/train_config.yaml \
  --checkpoint /path/to/best.pth \
  --output-dir ./outputs

# Inference on a video file
python scripts/inference.py \
  --video /path/to/video.mp4 \
  --instruction "Walk along the corridor and turn right at the door" \
  --checkpoint /path/to/best.pth
```

### Evaluation

```bash
python scripts/evaluate.py \
  --config configs/train_config.yaml \
  --checkpoint /path/to/best.pth \
  --split val_unseen \
  --save-vis
```

**Evaluation Metrics:**

| Category | Metric | Description |
|:---------|:-------|:------------|
| **Heatmap** | Peak Error | Peak position error (pixels) |
| | IoU@0.1/0.3/0.5 | Multi-threshold Intersection over Union |
| **Trajectory** | ADE | Average Displacement Error |
| | FDE | Final Displacement Error |
| **Progress** | MAE | Mean Absolute Error |

### Visualization

```bash
# Heatmap visualization (4-view panoramic)
python scripts/visualize_heatmap.py --checkpoint /path/to/best.pth --num-samples 10

# Trajectory + BEV visualization
python scripts/visualize_trajectory_heatmaps.py --checkpoint /path/to/best.pth
```

---

## Model Architecture

### Overall Architecture

```
+------------------------------------------------------------------------------+
|                      VLN Pipeline (Qwen3.5-9B)                               |
+------------------------------------------------------------------------------+
|                                                                              |
|  Inputs                                                                      |
|  +----------------+  +----------------+  +----------------+                  |
|  | History Frames |  | Current Frame  |  |  Instruction   |                  |
|  | (N panoramas)  |  | (4 dirs, 256^2)|  |    (text)      |                  |
|  +-------+--------+  +-------+--------+  +-------+--------+                 |
|          +-------------------+-------------------+                           |
|                              v                                               |
|                    +------------------+                                       |
|                    |   Qwen3.5-9B     |  <- Frozen (optional LoRA)            |
|                    | (Vision + LLM)   |                                       |
|                    +--------+---------+                                       |
|                             |                                                |
|            +----------------+----------------+                               |
|            |                |                |                                |
|     ViT features      LLM features      Text hidden                         |
|     (16x16, multi-L)  (8x8, multi-L)    states                              |
|            |                |                |                                |
|     +------+------+  +-----+------+         |                                |
|     | DPT-Lite    |  | DPT-Lite   |         |                                |
|     | Fusion(ViT) |  | Fusion(LLM)|         |                                |
|     +------+------+  +-----+------+         |                                |
|            |               |                 |                                |
|            |         +-----+------+          |                                |
|            |         |  Coarse    |<---------+                                |
|            |         |  8x8 + vis |  query_proj(text) x fused_llm            |
|            |         +-----+------+                                          |
|            |               |                                                 |
|      +-----+---------------+---+                                             |
|      |    Fine Localization    |  ViT features + coarse heatmap + text       |
|      |    16x16 -> 64x64      |  ConvTranspose decoder                      |
|      +------------+-----------+                                              |
|                   |                                                          |
|            +------+------+                                                   |
|            |  Heatmaps   |  (N_hist, 4, 64, 64)                             |
|            | + Visibility |  (N_hist, 4)                                     |
|            +-------------+                                                   |
|                                                                              |
|  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -                |
|  LLM Projector: 4096 -> 1024                                                |
|            |                                                                 |
|   +--------+---------+                                                       |
|   v                  v                                                       |
|  +--------------+ +--------------+                                           |
|  | Trajectory   | |  Progress    |                                           |
|  | (Transformer | |    (MLP)     |                                           |
|  |  + DDPM)     | |              |                                           |
|  +------+-------+ +------+-------+                                           |
|         v                v                                                   |
|     [24, 3]           [0, 1]                                                 |
|    (x, y, theta)     progress                                                |
|                                                                              |
+------------------------------------------------------------------------------+
```

**Training Strategies:**

| Stage | Trainable Modules | Frozen Modules | Description |
|:------|:------------------|:---------------|:------------|
| `coarse_to_fine_64` | HeatmapVLN, TransformerActionHead, ProgressHead, LLM Projector | All of Qwen3.5 | ~2M trainable parameters |
| `heatmap_only_64` | HeatmapVLN, LLM Projector | Everything else | Heatmap-only training (lighter) |

**Prediction Head Outputs:**

| Head | Output Shape | Method |
|:-----|:-------------|:-------|
| Heatmap | (N_hist, 4, 64, 64) | Coarse-to-Fine (cosine sim -> ConvTranspose) |
| Visibility | (N_hist, 4) | MLP (query + coarse heatmap -> binary) |
| Trajectory | [B, 24, 3] | Transformer Decoder + DDPM |
| Progress | [B, 1] | 3-layer MLP regression |

### Heatmap Generation Module

Uses a **Coarse-to-Fine** two-stage architecture leveraging multi-layer intermediate features from the frozen Qwen3.5-9B backbone:

```
Qwen3.5-9B (frozen)
    |
    +-- ViT blocks [6, 12, 18, 24] -> 16x16 features (c_vit=1152)
    +-- LLM full_attention layers [7, 15, 23] -> 8x8 features (c_llm=4096)
    +-- Text hidden states -> query vector (c_llm=4096)
    |
    +-- DPT-Lite Fusion (ViT) -> 16x16, c_fused=256
    +-- DPT-Lite Fusion (LLM) -> 8x8, c_fused=256
    |
    v Coarse Localization (8x8)
    query_proj(text) x fused_llm -> cosine similarity -> 8x8 coarse heatmap
    concat(query, coarse_flat) -> MLP -> visibility logits
    |
    v Fine Localization (64x64)
    FiLM modulation: query -> c_fused, element-wise multiply with ViT features
    Spatial attention: coarse heatmap 8x8 -> 16x16 as attention prior
    ConvTranspose decoder: 16x16 -> 32x32 -> 64x64
    |
    v Output
    heatmaps: (N_hist, 4, 64, 64) -- sigmoid activation
    visibility: (N_hist, 4) -- logits, sigmoid-gated at inference
```

**Loss Design (v10):**

| Loss | Scope | Description |
|:-----|:------|:------------|
| **Visibility BCE** | All samples | `pos_weight=7.0` to correct class imbalance (neg/pos ~ 7:1) |
| **Softmax CE** | Visible views (~13%) | 4096-pixel classification; inter-pixel competition prevents false positives |
| **Neg BCE** | Invisible views (~87%) | Per-pixel push-to-zero; provides dense gradient signal |
| **Coord Loss** | Visible views | Soft-argmax coordinate error; low-weight auxiliary |

At inference, the output is gated as `softmax(logit(sigmoid_heatmap)) * sigmoid(visibility)`, ensuring train-inference semantic alignment.

### Trajectory Prediction Module

| Component | Output | Description |
|:----------|:-------|:------------|
| `TransformerActionHead` | (x, y, theta) x 24 | Transformer Decoder + DDPM (recommended) |
| `DiffusionActionHead` | (x, y, theta) x 24 | UNet1D + DDPM (legacy) |
| `ProgressPredictionHead` | [0, 1] | 3-layer MLP, replaces binary Stop Head |
| `StopPredictionHead` | binary | Focal Loss based (deprecated) |

---

## Dataset

### Data Format

Two storage formats are supported, automatically detected via the `storage_format` field in `meta.json`:

**Format 1: Frames (per-frame files)**

```
<data_root>/
+-- train/
|   +-- <scene_id>/
|       +-- clip_000000/
|           +-- meta.json             # Metadata
|           +-- poses.json            # T x 4x4 pose matrices
|           +-- intrinsics.json       # Camera intrinsics (optional)
|           +-- rgb/                  # RGB image sequence
|           |   +-- front/            # Panoramic: subdirectories per direction
|           |   |   +-- 000000.jpg
|           |   +-- right/
|           |   +-- back/
|           |   +-- left/
|           +-- depth/                # Depth maps (optional, for occlusion)
|           |   +-- 000000.npy
|           +-- actions.npy           # Continuous actions [T, 2] (dx, dy)
|           +-- discrete_actions.npy  # Discrete actions [T]
+-- val_unseen/
    +-- ...
```

**Format 2: Chunks (chunked NPZ, recommended)**

```
<data_root>/
+-- train/
|   +-- <scene_id>/
|       +-- clip_000000/
|           +-- meta.json
|           +-- intrinsics.json
|           +-- chunks/
|           |   +-- chunk_000000.npz
|           |   +-- chunk_000001.npz
|           +-- actions.npy
|           +-- discrete_actions.npy
+-- val_unseen/
    +-- ...
```

**meta.json Fields:**

| Field | Type | Description |
|:------|:-----|:------------|
| `num_frames` | int | Total number of frames T |
| `instruction` | str | Navigation instruction text |
| `trajectory_id` | int | Trajectory ID (for FGR2R sub-instruction matching) |
| `storage_format` | str | `"frames"` or `"chunks"` |

### Sampling Strategies

| Strategy | Data Diversity | Description |
|:---------|:--------------:|:------------|
| Sliding Window | Low | Fixed-stride traversal |
| Clip-level | Medium | Random sampling per clip |
| **Random Subsequence** | High | Dynamic subsequence + sub-instructions (recommended) |

<details>
<summary>Random Subsequence Sampling Details</summary>

Each epoch generates different subsequences from the same clip, greatly increasing data diversity:

```
Original Clip: [Frame 0, Frame 1, ..., Frame 99]

Subsequence 1: [10, 50]  -> progress: 0% -> 100%
Subsequence 2: [30, 80]  -> progress: 0% -> 100%
Subsequence 3: [5, 70]   -> progress: 0% -> 100%
```

**Configuration:**

```yaml
data:
  trajectory:
    random_subsequence: true
    min_subsequence_length: 30
    subsequence_samples_per_clip: 5
    samples_per_clip: 30
    use_subinstruction: true
    enable_trajectory_augmentation: true
```

**Data volume calculation:**

```
Per epoch = clips x subseq x samples = 1000 x 5 x 30 = 150,000 samples
```

</details>

---

## Configuration

Main config files: `configs/train_config.yaml` (full training) or `configs/train_heatmap_config.yaml` (heatmap-only training)

<details>
<summary>Full Configuration Example</summary>

```yaml
# Model configuration
model:
  type: vln_pipeline

  # Qwen3.5 configuration
  llm:
    model_path: ./models/qwen_3.5
    hidden_dim: 4096
    token_dim: 1024
    torch_dtype: bfloat16
    attn_implementation: sdpa       # or flash_attention_2
    enable_packing: false
    max_seq_length: 8192
    spatial_merge_size: 2

    # LoRA fine-tuning (optional, disabled by default)
    use_lora: false
    lora_rank: 16
    lora_alpha: 32
    lora_num_layers: 4

  # HeatmapVLN v2 Coarse-to-Fine configuration
  heatmap:
    enable: true
    c_vit: 1152                     # ViT hidden dimension (Qwen3.5)
    c_llm: 4096                     # LLM hidden dimension
    c_fused: 256                    # DPT-Lite fusion output dim
    vit_layer_indices: [6, 12, 18, 24]    # ViT blocks to hook
    llm_layer_indices: [7, 15, 23]        # LLM full_attention layers
    heatmap_size: [64, 64]
    # Loss weights
    lambda_vis: 1.0
    lambda_coord: 0.2
    lambda_peak: 1.0

  # Transformer action head configuration
  action_head:
    enable: true
    type: transformer
    transformer:
      vlm_token_dim: 1024
      n_emb: 384
      predict_size: 24              # 24-step prediction
      n_layer: 16
      n_head: 6
      n_cond_layers: 4
      action_dim: 3                 # (x, y, theta)
      num_train_timesteps: 20
      causal_attn: true

  # Progress prediction head
  progress_head:
    enable: true
    hidden_dim: 512
    dropout: 0.3

  # Stop prediction head (deprecated)
  stop_head:
    enable: false

# Training strategy
training:
  stages:
    - name: coarse_to_fine_64
      epochs: 10
      hm_size: [64, 64]
      heatmap_loss_type: heatmap_vln
      train_heatmap: true
      train_action: true
      trainable_modules:
        - heatmap_vln
        - transformer_action_head
        - progress_head
        - llm_projector
      frozen_modules:
        - qwen3_5

# Optimizer configuration
optim:
  optimizer: adamw
  learning_rate: 2.0e-4
  heatmap_lr: 2.0e-4
  transformer_action_lr: 1.0e-4
  progress_lr: 1.0e-4
  llm_projector_lr: 1.0e-4
  weight_decay: 1.0e-2
  grad_clip: 1.0
  amp: bf16
  scheduler: cosine
  warmup_ratio: 0.1
  batch_size: 4
  grad_accum_steps: 2               # effective batch = 8

# Loss weights
loss:
  heatmap_loss_type: heatmap_vln
  heatmap_vln:
    lambda_vis: 1.0
    lambda_coord: 0.2
    lambda_peak: 1.0
    lambda_neg: 1.0
    vis_pos_weight: 7.0
  heatmap_weight: 1.0
  trajectory_weight: 1.0
  progress_weight: 1.0
```

</details>

---

## Training Outputs

Each training run creates an independent `run_<timestamp>/` directory under `log.out_dir`, with a `latest` symlink pointing to the most recent run.

```
run_YYYYMMDD_HHMMSS/
+-- manifest/           # Config, args, git state, environment info, summary
+-- logs/
|   +-- train.log       # Training log
|   +-- metrics.jsonl   # Structured metrics stream
+-- checkpoints/
|   +-- epoch_001.pth
|   +-- best.pth
|   +-- latest.pth
+-- visualizations/     # Train/val heatmap visualizations
+-- plots/              # Training curve plots
+-- tensorboard/        # TensorBoard entry point
```

See [Training Outputs Documentation](docs/training_outputs.md) for details.

---

## FAQ

<details>
<summary><b>Out of Memory (CUDA OOM)</b></summary>

Reduce batch size and increase gradient accumulation:

```yaml
optim:
  batch_size: 2
  grad_accum_steps: 16  # effective batch = 32
```

</details>

<details>
<summary><b>Heatmap is all black</b></summary>

Check `diag/pred_heatmap_max` in TensorBoard:
- If < 0.1, the heatmap has collapsed
- Verify the visibility head is working correctly: `vis_pos_weight` should be set to 7.0 to correct class imbalance
- Ensure neg_loss weight is not too high, which can suppress the positive sample gradient signal

</details>

<details>
<summary><b>How to resume training?</b></summary>

```bash
python scripts/train.py --config configs/train_config.yaml --auto-resume
```

</details>

<details>
<summary><b>How to enable/disable specific prediction heads?</b></summary>

```yaml
model:
  heatmap:
    enable: true           # Heatmap

  action_head:
    enable: true           # Trajectory prediction
    type: transformer      # transformer (recommended) or legacy

  progress_head:
    enable: true           # Progress prediction

  stop_head:
    enable: false          # Stop prediction (deprecated, use progress instead)
```

</details>

<details>
<summary><b>How to configure LoRA fine-tuning?</b></summary>

```yaml
model:
  llm:
    use_lora: true
    lora_rank: 16
    lora_alpha: 32
    lora_num_layers: 4      # Last 4 layers

optim:
  lora_lr: 1.0e-5          # Use very low learning rate for LoRA
```

</details>

---

## Project Structure

```
HeatmapVLN/
+-- configs/
|   +-- train_config.yaml              # Full training config (heatmap+trajectory+progress)
|   +-- train_heatmap_config.yaml      # Heatmap-only training config
|   +-- train_heatmap_config_2.yaml    # Heatmap config variant
+-- scripts/
|   +-- train.py                       # Training script
|   +-- inference.py                   # Inference script
|   +-- evaluate.py                    # Evaluation script
|   +-- eval_heatmap.py                # Heatmap-specific evaluation
|   +-- visualize_heatmap.py           # Heatmap visualization (4-view panoramic)
|   +-- visualize_trajectory_heatmaps.py  # Trajectory + BEV visualization
|   +-- compute_action_stats.py        # Dataset action statistics
+-- src/
|   +-- data/                          # Data loading
|   |   +-- vln_sliding_window_dataset.py  # Sliding window + trajectory dataset
|   |   +-- tokenized_dataset.py       # Qwen3.5 tokenization
|   |   +-- packing_collator.py        # Sequence packing collator
|   |   +-- panoramic_tokenized_collator.py  # Panoramic multi-view collator
|   +-- models/                        # Model definitions
|   |   +-- pipeline.py                # VLNPipeline main module
|   |   +-- qwen3_5/                   # Qwen3.5 integration (with LoRA support)
|   |   |   +-- integration.py         # Model loading & forward pass
|   |   |   +-- sequence_packing.py    # Sequence packing utilities
|   |   +-- heatmap/                   # Heatmap module (Coarse-to-Fine)
|   |   |   +-- heatmap_vln.py         # HeatmapVLN complete model
|   |   |   +-- heatmap_vln_loss.py    # Multi-component loss functions
|   |   |   +-- feature_extractor.py   # Hook-based feature extractor
|   |   |   +-- coarse_localization.py # Coarse localization (8x8 + visibility)
|   |   |   +-- fine_localization.py   # Fine localization (64x64)
|   |   |   +-- dpt_lite_fusion.py     # Multi-layer feature fusion
|   |   |   +-- input_constructor.py   # Input construction & text positioning
|   |   +-- action/                    # Action module
|   |       +-- transformer_action_head.py  # Transformer + DDPM (recommended)
|   |       +-- diffusion_action_head.py    # UNet1D + DDPM (legacy)
|   |       +-- progress_head.py       # Progress prediction head
|   |       +-- stop_head.py           # Stop prediction head (deprecated)
|   |       +-- action_config.py       # Action head configuration
|   |       +-- diffusion/             # 1D Diffusion submodule
|   |           +-- conditional_unet1d.py
|   |           +-- conv1d_components.py
|   |           +-- positional_embedding.py
|   +-- utils/                         # Utilities
|       +-- gpu_heatmap.py             # GPU heatmap computation
|       +-- loss.py                    # Navigation loss functions
|       +-- logger.py                  # Logging configuration
|       +-- notifier.py                # Feishu notification
|       +-- visualization.py           # Visualization utilities
|       +-- frame_vis_utils.py         # Frame visualization utilities
|       +-- html_template.py           # HTML report template
|       +-- path_utils.py              # Path utilities
|       +-- plotting_config.py         # Matplotlib configuration
+-- data/
|   +-- fgr2r/                         # FGR2R sub-instruction data
|       +-- subinstr_mapping.json.gz
+-- docker/                            # Docker deployment config
+-- docs/                              # Supplementary documentation
|   +-- loss.md                        # Loss design details
|   +-- heatmap_loss_strategy.md       # Heatmap loss strategy analysis
|   +-- training_outputs.md            # Training output directory documentation
+-- assets/
|   +-- architecture.png
+-- requirements.txt
+-- README.md
```

---

## Acknowledgements

- [Qwen3.5-VL](https://github.com/QwenLM/Qwen-VL) - Vision-language backbone model
- [InternNav](https://github.com/OpenRobotLab/InternNav) - Transformer Action Head reference implementation
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Diffusion policy reference

---

## License

This project is licensed under the [MIT License](LICENSE).
