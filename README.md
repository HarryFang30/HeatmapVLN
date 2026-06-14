<p align="center">
  <h1 align="center">HeatmapVLN</h1>
  <p align="center">
    <strong>Coarse-to-Fine Spatial Heatmap Generation for Vision-Language Navigation</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/release/python-3110/"><img alt="Python 3.11" src="https://img.shields.io/badge/python-3.11-blue.svg"></a>
    <a href="https://pytorch.org/"><img alt="PyTorch 2.7" src="https://img.shields.io/badge/pytorch-2.7.0-ee4c2c.svg"></a>
    <a href="https://huggingface.co/docs/transformers"><img alt="Transformers" src="https://img.shields.io/badge/transformers-4.51.0-yellow.svg"></a>
    <a href="https://developer.nvidia.com/cuda-toolkit"><img alt="CUDA 12.8" src="https://img.shields.io/badge/cuda-12.8-76b900.svg"></a>
  </p>
</p>

---

HeatmapVLN addresses the spatial grounding problem in Vision-Language Navigation by jointly predicting **history-aware spatial heatmaps** and **multi-step action trajectories** from panoramic observations. The system couples a frozen Qwen2.5-VL vision-language backbone with two specialized heads: a coarse-to-fine heatmap decoder (System 2) for spatial localization and a diffusion-based NextDiT trajectory generator (System 1) for action prediction.

<p align="center">
  <img src="assets/system2_achitecture.svg" alt="HeatmapVLN Architecture" width="85%">
</p>

## Overview

### Method

Given a sequence of panoramic observations (4 views per timestep) and a natural language instruction, HeatmapVLN:

1. **Encodes** the visual-linguistic input through a frozen Qwen2.5-VL backbone with selective LoRA adaptation (layers 5, 6, 12, 13, 19, 20);
2. **Extracts** multi-scale intermediate features from both the ViT encoder and the LLM decoder via layer-wise hooks;
3. **Predicts visibility and spatial heatmaps** through a coarse-to-fine pipeline:
   - *Coarse stage*: Trajectory-Guided Attention aggregates history-relative spatial cues with positional encoding;
   - *Fine stage*: DPT-Lite fusion combines ViT features (16 &times; 16), LLM features (8 &times; 8), and coarse attention output to produce 64 &times; 64 heatmaps;
4. **Generates 32-step trajectories** via a NextDiT denoiser conditioned on VLM hidden states and DINOv2 visual memory, trained with Flow Matching.

### Architecture

```
  Panoramic Input (4 views × (N_hist + 1)) + Instruction
                          │
                          ▼
         ┌────────────────────────────────┐
         │         Qwen2.5-VL            │
         │    (frozen + selective LoRA)    │
         │                                │
         │  ┌─────────┐   ┌───────────┐  │
         │  │   ViT    │   │    LLM    │  │
         │  │ features │   │ features  │  │
         │  │(16×16,4L)│   │(8×8, 3L)  │  │
         │  └────┬─────┘   └─────┬─────┘  │
         │       │               │         │
         └───────┼───────────────┼─────────┘
                 │               │
       ┌─────────┴───┐    ┌─────┴──────────┐
       │             │    │                │
       ▼             ▼    ▼                ▼
  ┌─────────────────────────┐   ┌──────────────────┐
  │      HeatmapVLN v2      │   │   NextDiT Sys.1  │
  │      (System 2)          │   │                  │
  │                          │   │  DINOv2 encoder  │
  │  DPT-Lite Fusion (ViT)  │   │  Memory Encoder  │
  │  DPT-Lite Fusion (LLM)  │   │  QFormer (32 q)  │
  │  Trajectory-Guided Attn  │   │  DiT (12 layers) │
  │  Fine Localization       │   │  Flow Matching   │
  └────────────┬─────────────┘   └────────┬─────────┘
               │                          │
               ▼                          ▼
      Visibility  (N, 4)         Trajectory  (B, 32, 3)
      Heatmap  (N, 4, 64, 64)   [Δx, Δy, Δheading]
```

### Component Summary

| Component | Role | Parameters | Training |
|:----------|:-----|:-----------|:---------|
| Qwen2.5-VL | Vision-language backbone (InternNav) | 3.2B (frozen) | LoRA only (~2M) |
| HeatmapVLN v2 | Coarse-to-fine spatial heatmap prediction | ~2M | Stage 1 |
| NextDiT System 1 | Diffusion trajectory prediction (32 steps) | ~51M total | ~24M selective in Stage 2 |

### Loss Functions

| Loss | Target | Description |
|:-----|:-------|:------------|
| Visibility BCE | `vis_head` | Weighted binary classification of history-view visibility |
| Peak Cross-Entropy | Heatmap backbone | Softmax pixel classification on visible-view heatmaps |
| Coordinate Loss | Heatmap backbone | Soft-argmax peak position refinement (auxiliary) |
| Negative BCE | Heatmap backbone | Pushes invisible-view heatmaps toward zero |
| Flow Matching MSE | NextDiT | Velocity field prediction for trajectory denoising |

## Getting Started

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| Python | 3.11 |
| PyTorch | 2.7.0 |
| CUDA | 12.8 |
| GPU VRAM | &ge; 48 GB (A6000 / A100) |

### Installation

```bash
conda create -n heatmapvln python=3.11 -y && conda activate heatmapvln

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

> The default attention backend is SDPA. Flash Attention is optional and requires a compatible GLIBC environment. Sequence packing is disabled by default.

### Model Weights

By default the training configs load the unified InternNav checkpoint directly:

| Weight | Default Path | Description |
|:-------|:-------------|:------------|
| InternNav Full Model | `$INTERNNAV_MODEL_PATH` | Qwen2.5-VL backbone + NextDiT System 1 + `rgb_model` DINOv2/DepthAnything encoder |

Set `paths.internnav_model_path` in YAML, or export `INTERNNAV_MODEL_PATH=/path/to/InternNav_Model`,
to choose the unified checkpoint directory without editing code. The bundled launch scripts default this
environment variable to `/workspace/InternNav_Model` if it is not already set.

The old split layout is still supported for compatibility. To extract backbone
and System 1 from a unified InternNav checkpoint:

```bash
python scripts/tools/convert_internnav_backbone.py \
    --src /path/to/InternNav_Model \
    --backbone-dst models/internnav_backbone \
    --system1-dst models/internnav_system1.safetensors
```

## Dataset

The data pipeline supports panoramic 4-view navigation recordings organized as clips. Two directory layouts are accepted:

**Split-based** (recommended):
```
<root>/train/<scene_id>/clip_000000/
<root>/val_unseen/<scene_id>/clip_000000/
```

**Flat** (auto-split by scene hash):
```
<root>/<scene_id>/clip_000000/
```

<details>
<summary><strong>Clip directory structure</strong></summary>

Each clip contains either per-frame files or packed chunks:

```
clip_xxxxxx/
├── meta.json                   # metadata (storage_format, num_frames, ...)
├── poses.json                  # camera poses per frame
├── intrinsics.json             # camera intrinsics (optional)
├── rgb/
│   ├── front/                  # panoramic views
│   ├── right/
│   ├── back/
│   └── left/
├── depth/                      # depth maps (optional)
├── actions.npy                 # agent-local displacements [frame_i → frame_i+1]
└── discrete_actions.npy        # {0: STOP, 1: FORWARD, 2: LEFT, 3: RIGHT}
```

Alternatively, frames may be stored as `chunks/chunk_*.npz` (auto-detected).

</details>

FGR2R sub-instruction support requires `data.trajectory.use_subinstruction: true` and `data/fgr2r/subinstr_mapping.json.gz`.

## Training

### Two-Stage Training Strategy

The default pipeline follows a two-stage curriculum:

| | Stage 1: Spatial Grounding | Stage 2: Bridge Adaptation |
|:---|:---|:---|
| **Config** | `train_heatmap_config.yaml` | `train_config_internnav.yaml` |
| **Objective** | Train LoRA + HeatmapVLN for panoramic spatial understanding | Connect frozen panoramic System2 features to frozen InternNav System1 |
| **Trainable** | LoRA, HeatmapVLN, vis_head, llm_projector | `latent_queries`, `cond_projector` |
| **Frozen** | VLM base weights, NextDiT | VLM/LoRA/HeatmapVLN, DINOv2, memory_encoder, DiT, action enc/dec |
| **Learning Rate** | 5e-5 (LoRA), 2e-4 (vis_head) | 1e-4 (bridge) |

### Commands

```bash
# Stage 1: Heatmap pretraining
python scripts/run.py train --config configs/train_heatmap_config.yaml

# Stage 2: Trajectory fine-tuning (load Stage 1 weights)
python scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --load-weights /path/to/stage1/best.pth

# Resume training
python scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --load-weights /path/to/stage1/best.pth \
    --auto-resume

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --load-weights /path/to/stage1/best.pth \
    --distributed
```

### Quick Validation

```bash
# Dry run (build model + data, no training)
python scripts/run.py train \
    --config configs/train_config_internnav.yaml \
    --load-weights /path/to/stage1/best.pth \
    --dry-run

# Smoke test (2 batches, 1 epoch)
python scripts/run.py train --config configs/train_config_internnav.yaml \
    --load-weights /path/to/stage1/best.pth \
    --epochs 1 --max-batches 2
```

## Evaluation

```bash
# General evaluation
python scripts/run.py evaluate \
    --checkpoint /path/to/best.pth --split val_unseen --save-vis

# Heatmap-specific metrics
python scripts/run.py evaluate heatmap \
    --checkpoint /path/to/best.pth --max-samples 200

# R2R val_unseen (Habitat)
python scripts/run.py evaluate r2r \
    --base_checkpoint /path/to/stage1/best.pth \
    --checkpoint /path/to/stage2_bridge/best.pth
```

## Inference & Visualization

```bash
# Single-clip inference
python scripts/run.py inference \
    --checkpoint /path/to/best.pth \
    --clip /path/to/clip_dir \
    --instruction "Walk past the kitchen island and stop at the door" \
    --output-dir ./outputs

# 4-view heatmap visualization
python scripts/run.py visualize heatmap \
    --checkpoint /path/to/best.pth --num-samples 10

# Trajectory sequence visualization
python scripts/run.py visualize trajectory \
    --checkpoint /path/to/best.pth --num-clips 3 --frames-per-clip 32
```

### Stage1-S2 Heatmap Impact Check

Stage1-S2 checkpoints fine-tune the Qwen LoRA weights only. They do **not**
continue training `heatmap_vln`, and the saved config usually has
`model.heatmap.enable: false`. Do not pass a Stage1-S2 checkpoint directly to
heatmap visualization and interpret it as a full heatmap model.

To check whether Stage1-S2 changed heatmap generation, build two visualization
checkpoints:

1. Stage1 baseline: Stage1 LoRA + Stage1 `heatmap_vln`.
2. Stage1+S2 overlay: Stage1 `heatmap_vln`, with Stage1-S2 Qwen LoRA keys
   overriding the matching Stage1 LoRA keys.

The following snippet also patches old checkpoint paths to the local model and
`/workspace/val_unseen` dataset root:

```bash
python - <<'PY'
from pathlib import Path
import copy
import torch

stage1_path = Path('checkpoints/stage1_latest.pth')
stage1_s2_path = Path('checkpoints/stage1-s2_latest.pth')
out_dir = Path('debug/topdown_trajectory_checkpoint_inputs')
out_dir.mkdir(parents=True, exist_ok=True)

stage1 = torch.load(stage1_path, map_location='cpu', weights_only=False)
stage1_s2 = torch.load(stage1_s2_path, map_location='cpu', weights_only=False)

def patch_config(ckpt):
    cfg = copy.deepcopy(ckpt['config'])
    cfg.setdefault('data', {})['root'] = '/workspace/val_unseen'
    cfg['data']['val_split'] = 'all'
    cfg.setdefault('model', {}).setdefault('llm', {})['model_path'] = 'models/internnav_backbone'
    cfg['model']['llm']['attn_implementation'] = 'sdpa'
    cfg['model']['llm']['gradient_checkpointing'] = False
    cfg['model'].setdefault('heatmap', {})['enable'] = True
    cfg['model'].setdefault('action_head', {})['enable'] = False
    cfg.setdefault('log', {})['enable_timing'] = False
    return cfg

base = copy.deepcopy(stage1)
base['config'] = patch_config(stage1)
base['stage_name'] = 'stage1_heatmap_visualization_config_patched'
base_out = out_dir / 'stage1_heatmap_visualization_config_patched.pth'
torch.save(base, base_out)

merged = copy.deepcopy(stage1)
merged['config'] = patch_config(stage1)
merged_sd = copy.deepcopy(stage1.get('trainable_state_dict', {}))
merged_sd.update(stage1_s2.get('trainable_state_dict', {}))
merged['trainable_state_dict'] = merged_sd
merged['stage_name'] = 'stage1_heatmap_plus_stage1_s2_lora_visualization'
merged['source_checkpoints'] = {
    'stage1_heatmap': str(stage1_path),
    'stage1_s2_lora_overlay': str(stage1_s2_path),
    'note': 'stage1 heatmap_vln plus stage1-s2 LoRA overlay',
}
merged_out = out_dir / 'stage1_heatmap_plus_stage1_s2_lora_visualization.pth'
torch.save(merged, merged_out)

print(base_out)
print(merged_out)
PY
```

Then run the top-down trajectory heatmap visualization on both checkpoints with
the same selection parameters:

```bash
# Stage1 baseline
CUDA_VISIBLE_DEVICES=1 python scripts/run.py visualize trajectory \
    --checkpoint debug/topdown_trajectory_checkpoint_inputs/stage1_heatmap_visualization_config_patched.pth \
    --data-root /workspace/val_unseen \
    --split all \
    --num-clips 2 \
    --frames-per-clip 12 \
    --output-dir debug/topdown_trajectory_vis_stage1 \
    --device cuda:0 \
    --attn-impl sdpa \
    --tile-size 72

# Stage1 heatmap + Stage1-S2 Qwen LoRA overlay
CUDA_VISIBLE_DEVICES=1 python scripts/run.py visualize trajectory \
    --checkpoint debug/topdown_trajectory_checkpoint_inputs/stage1_heatmap_plus_stage1_s2_lora_visualization.pth \
    --data-root /workspace/val_unseen \
    --split all \
    --num-clips 2 \
    --frames-per-clip 12 \
    --output-dir debug/topdown_trajectory_vis_stage1_plus_stage1_s2 \
    --device cuda:0 \
    --attn-impl sdpa \
    --tile-size 72
```

Important notes:

- `--split all` is intentional for `/workspace/val_unseen`, which has scene
  directories directly under the root rather than a `val/` subdirectory.
- Use `--attn-impl sdpa` on machines without `flash_attn` installed.
- `scripts/visualization/trajectory_heatmaps.py` must pass
  `history_rel_poses` into the model. Stage1 uses `TrajectoryGuidedAttention`;
  if this tensor is omitted, the trajectory token is zeroed and predictions can
  collapse to a fixed view (commonly the Back view), producing misleading
  visualizations.
- The current `/workspace/val_unseen` metadata does not include `instruction`,
  so these visualizations test the visual/trajectory-conditioned heatmap path
  with empty instruction text. Use a dataset whose `meta.json` includes
  `instruction` for instruction-conditioned checks.

## Training Outputs

Each run produces an isolated, reproducible output directory:

```
<out_dir>/run_YYYYMMDD_HHMMSS/
├── manifest/          # frozen config, git SHA, environment snapshot
├── logs/              # train.log, metrics.jsonl (step-level)
├── checkpoints/       # epoch_*.pth, best.pth, latest.pth
├── visualizations/    # train/ and val/ heatmap renders
├── plots/             # loss curves, learning rate schedule
└── tensorboard/       # event files
```

A `latest` symlink always points to the most recent run. Monitor via:

```bash
tensorboard --logdir <tensorboard_dir> --port 6006
```

## Repository Structure

```
HeatmapVLN/
├── configs/                        # YAML training configurations
├── scripts/
│   ├── run.py                      # Unified CLI: train / evaluate / visualize / inference
│   ├── train.py                    # Training entrypoint
│   ├── evaluate.py                 # Evaluation entrypoint
│   ├── visualize.py                # Visualization entrypoint
│   ├── inference.py                # Inference entrypoint
│   ├── training/                   # Training loop, optimizer, checkpointing, EMA, DDP
│   ├── evaluation/                 # General, heatmap, and R2R evaluation
│   ├── visualization/              # Heatmap and trajectory rendering
│   ├── tools/                      # Weight conversion, action statistics
│   └── ops/                        # GPU monitoring with Feishu webhook alerts
├── src/
│   ├── data/                       # VLNSlidingWindowDataset, collators, tokenization
│   ├── models/
│   │   ├── pipeline.py             # VLNPipeline: top-level model assembly
│   │   ├── qwen2_5_vl/            # Qwen2.5-VL integration, LoRA, sequence packing
│   │   ├── heatmap/               # HeatmapVLN v2: DPT-Lite, coarse/fine localization
│   │   └── action/                # NextDiT: DiT denoiser, DINOv2, Flow Matching
│   └── utils/                      # Logging, Feishu notifier, visualization, loss
├── models/                         # Pretrained weight directory
├── data/fgr2r/                     # FGR2R sub-instruction data
├── docs/                           # Design documents
└── docker/                         # Dockerfile, docker-compose, launch script
```

## Docker

```bash
docker build -f docker/Dockerfile -t heatmapvln:latest .

docker run --gpus all -it --rm \
    -v /path/to/data:/workspace/r2r_panoramic_data \
    -v /path/to/weights:/workspace/HeatmapVLN/models \
    --shm-size 8g -p 6006:6006 \
    heatmapvln:latest
```

An interactive launcher is also available: `./docker/docker-run.sh`

## Documentation

| Document | Description |
|:---------|:------------|
| [`docs/loss.md`](docs/loss.md) | Loss function components and weighting strategy |
| [`docs/heatmap_loss_strategy.md`](docs/heatmap_loss_strategy.md) | Temperature scheduling and heatmap loss design |
| [`docs/training_outputs.md`](docs/training_outputs.md) | Output directory structure and metrics format |
| [`docs/troubleshooting-guide.md`](docs/troubleshooting-guide.md) | Common issues and debugging guide |
| [`docs/ReadBeforeEvaluatingHabitat.md`](docs/ReadBeforeEvaluatingHabitat.md) | Habitat R2R evaluation setup notes |

## Acknowledgements

This project builds upon the following works:

- **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)** &mdash; Vision-language foundation model
- **[InternNav](https://github.com/)** &mdash; Navigation-tuned Qwen2.5-VL backbone and NextDiT System 1
- **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)** &mdash; Monocular depth estimation (DINOv2-vits)
- **[FGR2R](https://github.com/YicongHong/Fine-Grained-R2R)** &mdash; Fine-grained sub-instruction annotations for R2R (see [`data/fgr2r/LICENSE`](data/fgr2r/LICENSE))

## Citation

If you find this work useful, please consider citing:

```bibtex
@software{heatmapvln2026,
  author       = {Jialei Fang},
  title        = {HeatmapVLN: Coarse-to-Fine Spatial Heatmap Generation for Vision-Language Navigation},
  year         = {2026},
  url          = {https://github.com/HarryFang30/HeatmapVLN}
}
```
