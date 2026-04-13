# HeatmapVLN

**Coarse-to-Fine Heatmap Generation for Vision-Language Navigation**

HeatmapVLN is a training and evaluation framework for Vision-Language Navigation (VLN) that combines spatial heatmap prediction with trajectory generation. It leverages a frozen Qwen2.5-VL backbone (fine-tuned via LoRA) to jointly predict where the agent has been (history heatmaps) and where it should go next (future trajectories).

<p align="center">
  <img src="assets/system2_achitecture.svg" alt="HeatmapVLN System 2 Architecture" width="90%">
</p>

## Architecture

The pipeline consists of three core components:

```
Panoramic input (4 views × N history + current) + Text instruction
    │
    ▼
┌─────────────────────────────────────────────┐
│  Qwen2.5-VL  (frozen weights + selective    │
│               LoRA on layers 5,6,12,13,     │
│               19,20)                        │
│                                             │
│  Outputs:                                   │
│   • ViT intermediate features (multi-layer) │
│   • LLM intermediate features (multi-layer) │
│   • Text-anchor hidden states               │
│   • Trajectory hidden states (n_query=4)    │
└──────────┬──────────────────────┬───────────┘
           │                      │
    ┌──────▼──────┐       ┌──────▼──────────┐
    │ HeatmapVLN  │       │ NextDiT         │
    │ (System 2)  │       │ (System 1)      │
    │             │       │                 │
    │ Coarse:     │       │ DINOv2 encoder  │
    │  Trajectory │       │ + DiT denoiser  │
    │  Guided     │       │ + Flow Matching │
    │  Attention  │       │                 │
    │ Fine:       │       │                 │
    │  DPT-Lite   │       │                 │
    │  Fusion     │       │                 │
    └──────┬──────┘       └──────┬──────────┘
           │                      │
           ▼                      ▼
    Visibility (N,4)        Trajectory (B,T,3)
    Heatmap (N,4,64,64)     [dx, dy, dheading]
```

| Component | Description | Trainable Params |
|-----------|-------------|-----------------|
| **Qwen2.5-VL** | Vision-language backbone (InternNav) | LoRA only (~2M) |
| **HeatmapVLN v2** | Coarse-to-fine history heatmap prediction with DPT-Lite fusion | ~2M |
| **NextDiT System 1** | Diffusion-based trajectory prediction (32 steps) from InternNav | ~24M (selective) |

## Repository Structure

```
HeatmapVLN/
├── configs/                          # Training YAML configurations
│   ├── train_config_internnav.yaml   # Default: Stage 2 selective fine-tuning
│   ├── train_heatmap_config.yaml     # Stage 1: heatmap-only training
│   └── ...
├── scripts/
│   ├── run.py                        # Unified CLI entrypoint
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation (general / heatmap / R2R)
│   ├── visualize.py                  # Visualization (heatmap / trajectory)
│   ├── inference.py                  # Single-clip inference
│   ├── training/                     # Training loop, optimizer, checkpointing, etc.
│   ├── evaluation/                   # Evaluation subroutines
│   ├── visualization/                # Visualization subroutines
│   ├── tools/                        # Weight conversion, action statistics
│   └── ops/                          # GPU monitoring, Feishu notifications
├── src/
│   ├── data/                         # Dataset, collators, tokenization
│   ├── models/
│   │   ├── pipeline.py               # VLNPipeline (top-level assembly)
│   │   ├── qwen2_5_vl/              # Qwen2.5-VL integration & LoRA
│   │   ├── heatmap/                  # HeatmapVLN v2 (coarse-to-fine)
│   │   └── action/                   # NextDiT action head, diffusion modules
│   └── utils/                        # Logging, notifications, visualization
├── models/                           # Local model weights directory
├── docs/                             # Design documents, loss strategy, troubleshooting
├── docker/                           # Dockerfile, docker-compose, launch script
└── data/fgr2r/                       # FGR2R sub-instruction data & license
```

## Requirements

- Python 3.11
- PyTorch 2.7.0 with CUDA 12.8
- transformers 4.51.0
- 1+ NVIDIA GPU with >= 48 GB VRAM (A6000, A100, etc.)

## Installation

```bash
conda create -n heatmapvln python=3.11 -y
conda activate heatmapvln

# Install PyTorch with CUDA 12.8
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note:** The default attention implementation is `sdpa`. Flash Attention is optional and only recommended in dedicated environments with compatible GLIBC. Sequence packing is currently disabled (`model.llm.enable_packing: false`).

## Weight Preparation

The default configuration requires three sets of pretrained weights:

| Weight | Path | Source |
|--------|------|--------|
| InternNav Backbone | `models/internnav_backbone/` | Qwen2.5-VL fine-tuned by InternNav |
| System 1 Weights | `models/internnav_system1.safetensors` | NextDiT trajectory head |
| Depth Anything V2 | `models/depth_anything_v2_metric_hypersim_vits.pth` | DINOv2-vits depth encoder |

If you have the original InternNav model directory, split it into backbone and System 1 weights:

```bash
python scripts/tools/convert_internnav_backbone.py \
    --src /path/to/InternNav_Model \
    --backbone-dst models/internnav_backbone \
    --system1-dst models/internnav_system1.safetensors
```

The Depth Anything V2 checkpoint must be obtained separately and placed at the configured path.

## Dataset Format

The dataset loader (`src/data/vln_sliding_window_dataset.py`) supports two storage layouts:

### Split-based Layout (recommended)

```
<data_root>/
├── train/
│   └── <scene_id>/
│       └── clip_000000/
└── val_unseen/
    └── <scene_id>/
        └── clip_000000/
```

### Flat Layout (auto-split by scene hash)

```
<data_root>/
└── <scene_id>/
    └── clip_000000/
```

### Clip Structure

Each clip directory contains navigation data in either **frame** or **chunk** format:

<details>
<summary><b>Frame format</b></summary>

```
clip_xxxxxx/
├── meta.json
├── poses.json
├── intrinsics.json             # optional
├── rgb/
│   ├── front/                  # panoramic 4-view
│   ├── right/
│   ├── back/
│   └── left/
├── depth/                      # optional
├── actions.npy                 # (N,) agent-local displacements
└── discrete_actions.npy        # 0=STOP, 1=FWD, 2=LEFT, 3=RIGHT
```

</details>

<details>
<summary><b>Chunk format</b></summary>

```
clip_xxxxxx/
├── meta.json
├── intrinsics.json             # optional
├── chunks/
│   └── chunk_*.npz             # batched frames
├── actions.npy
└── discrete_actions.npy
```

</details>

To enable FGR2R sub-instructions, set `data.trajectory.use_subinstruction: true` in the config and provide `data/fgr2r/subinstr_mapping.json.gz`.

## Usage

All commands are accessible through the unified entrypoint `scripts/run.py`:

```bash
python scripts/run.py <command> [options]
```

### Quick Validation

```bash
# Dry run: build model and data pipeline without training
python scripts/run.py train --config configs/train_config_internnav.yaml --dry-run

# Smoke test: run 2 batches for 1 epoch
python scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --epochs 1 --max-batches 2
```

### Training

**Stage 1 — Heatmap pretraining** (train spatial understanding via LoRA + HeatmapVLN):

```bash
python scripts/run.py train --config configs/train_heatmap_config.yaml
```

**Stage 2 — Selective fine-tuning** (adapt System 1 trajectory head to the LoRA-modified backbone):

```bash
python scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --load-weights /path/to/stage1/checkpoints/best.pth
```

**Resume from checkpoint:**

```bash
python scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --auto-resume
```

**Multi-GPU training (DDP):**

```bash
torchrun --nproc_per_node=2 scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --distributed
```

### Evaluation

```bash
# General evaluation
python scripts/run.py evaluate \
    --config configs/train_config_internnav.yaml \
    --checkpoint /path/to/best.pth \
    --split val_unseen --save-vis

# Heatmap-specific evaluation
python scripts/run.py evaluate heatmap \
    --config configs/train_heatmap_config.yaml \
    --checkpoint /path/to/best.pth \
    --max-samples 200

# R2R val_unseen evaluation
python scripts/run.py evaluate r2r \
    --config configs/train_config_internnav.yaml \
    --checkpoint /path/to/best.pth
```

### Inference

```bash
# Single video
python scripts/run.py inference \
    --config configs/train_config_internnav.yaml \
    --checkpoint /path/to/best.pth \
    --video /path/to/video.mp4 \
    --instruction "Go forward and turn right at the door" \
    --output-dir ./outputs

# Single clip directory
python scripts/run.py inference \
    --config configs/train_config_internnav.yaml \
    --checkpoint /path/to/best.pth \
    --clip /path/to/clip_dir \
    --output-dir ./outputs
```

### Visualization

```bash
# 4-view heatmap visualization
python scripts/run.py visualize heatmap \
    --checkpoint /path/to/best.pth \
    --num-samples 10 \
    --output-dir ./vis_heatmap

# Temporal trajectory heatmap visualization
python scripts/run.py visualize trajectory \
    --checkpoint /path/to/best.pth \
    --num-clips 3 --frames-per-clip 32 \
    --output-dir ./vis_trajectory
```

## Training Strategy

The default two-stage training strategy (see `configs/`):

| Stage | Config | Trainable | Frozen |
|-------|--------|-----------|--------|
| **1. Heatmap** | `train_heatmap_config.yaml` | LoRA, HeatmapVLN, vis_head, llm_projector | VLM base, NextDiT |
| **2. Trajectory** | `train_config_internnav.yaml` | cond_projector, latent_queries, traj_dit, action_enc/dec | VLM base, LoRA, DINOv2, memory_encoder, rgb_resampler |

Stage 2 uses a layered learning rate schedule:
- Bridge layers (`cond_projector`, `latent_queries`): `1e-4`
- Trajectory DiT: `5e-5` (conservative, pretrained initialization)

## Training Outputs

Each run creates an isolated directory under `log.out_dir` with a `latest` symlink:

```
run_YYYYMMDD_HHMMSS/
├── manifest/          # config snapshot, git state, environment info
├── logs/              # train.log, metrics.jsonl
├── checkpoints/       # epoch_XXX.pth, best.pth, latest.pth
├── visualizations/    # train/ and val/ heatmap renders
├── plots/             # training curves
└── tensorboard/       # TensorBoard event files
```

Monitor training:

```bash
tensorboard --logdir /root/tf-logs --port 6006
```

## Docker

A Docker setup is available in `docker/`:

```bash
# Build
docker build -f docker/Dockerfile -t heatmapvln:latest .

# Interactive session
docker run --gpus all -it --rm \
    -v /path/to/data:/workspace/r2r_panoramic_data \
    -v /path/to/models:/workspace/HeatmapVLN/models \
    --shm-size 8g -p 6006:6006 \
    heatmapvln:latest

# Or use the interactive launcher
./docker/docker-run.sh
```

## GPU Monitoring

The repository includes a GPU idle monitoring script with Feishu (Lark) webhook notifications:

```bash
python scripts/ops/monitor_gpu_idle.py \
    --gpus 0,1 \
    --duration-sec 60 \
    --interval-sec 5
```

When idle GPUs are detected, it can optionally auto-launch training to occupy them.

## Documentation

Additional design and operational documentation is available in `docs/`:

| Document | Description |
|----------|-------------|
| `docs/loss.md` | Loss function design and component breakdown |
| `docs/heatmap_loss_strategy.md` | Heatmap loss scheduling and temperature strategy |
| `docs/training_outputs.md` | Training output directory structure |
| `docs/troubleshooting-guide.md` | Common issues and solutions |
| `docs/ReadBeforeEvaluatingHabitat.md` | Notes on Habitat-based R2R evaluation |

## License

This project builds upon the following works:
- [InternNav](https://github.com/) — Qwen2.5-VL backbone and System 1 trajectory head
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) — DINOv2-based depth encoder
- [FGR2R](https://github.com/) — Fine-grained R2R sub-instruction annotations (see `data/fgr2r/LICENSE`)
