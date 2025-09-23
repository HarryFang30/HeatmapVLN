# VLN Spatial-MLLM Pipeline

**First-Person Inter-Frame Heatmap Generation for Vision-Language Navigation**

This project implements a state-of-the-art VLN (Vision-Language Navigation) pipeline that generates first-person-view heatmaps demonstrating spatial relationships between video frames. The system uses a dual-encoder architecture with space-aware frame sampling to understand and visualize inter-frame spatial connections.

## 🎯 Project Goal

Generate **first-person-view heatmaps** that display spatial relationships between video frames, demonstrating the model's understanding of inter-frame spatial connections. The system answers the key question: *"When processing Frame A, where would the content from Frames B, C, D... appear if visible from Frame A's first-person perspective?"*

## 🏗️ Architecture Overview

### Dual-Encoder Pipeline: N_m → N_k → Heatmaps

```
📹 Video Input (N_m frames)
    ↓
🔍 VGGT (3D Encoder) → Geometry Extraction → Space-aware Sampling
    ↓                                              ↓
📐 Camera Poses + Depth Maps              🎯 Select N_k Keyframes
    ↓                                              ↓
🖼️ DINOv3 (2D Encoder) ← Index Selection ← Selected Frames
    ↓                           ↓
🔗 Feature Fusion: 3D Geometry + 2D Semantics
    ↓
🧠 Spatial-MLLM: LLM + Spatial Reasoning
    ↓
🗺️ First-Person Inter-Frame Heatmaps
```

### Key Components

- **VGGT (3D Path)**: Processes ALL frames for geometry extraction and space-aware sampling
- **DINOv3 (2D Path)**: Processes ONLY selected keyframes for rich semantic features  
- **Space-aware Sampling**: Intelligently selects N_k most informative frames from N_m total
- **Feature Fusion**: Combines 3D geometry with 2D semantics
- **Spatial-MLLM**: LLM-enhanced spatial reasoning for cross-frame understanding
- **Graph Upsampling**: Generates high-resolution heatmaps using ConvexUpSample

## 🚀 Quick Start

### 1. Environment Setup

1. **Create conda environment:**

```bash
conda create -n spatial-mllm python=3.10 -y
conda activate spatial-mllm
```

2. **Install core dependencies:**

```bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.3 accelerate==1.5.2 qwen_vl_utils decord
pip install flash-attn --no-build-isolation

# Additional dependencies for full functionality
pip install opencv-python matplotlib pillow numpy scipy
pip install wandb tqdm omegaconf pyyaml
pip install trimesh pyrender  # For 3D visualization
```

3. **Install project requirements:**

```bash
# Navigate to project directory
cd VLN/Project

# Install project-specific requirements
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Train the model
python main.py --config configs/default_config.yaml --mode train

# Run inference on a video
python main.py --config configs/default_config.yaml --mode inference --video_path /path/to/video.mp4

# Evaluate on benchmarks
python main.py --config configs/default_config.yaml --mode eval --benchmark VSI-Bench

# Preprocess data
python main.py --config configs/default_config.yaml --mode preprocess --input_dir /raw/data --output_dir /processed/data
```

## 📁 Project Structure

```
Project/
├── configs/                    # Configuration files
│   └── default_config.yaml    # Main configuration
├── src/                        # Source code
│   ├── data/                  # Data processing
│   │   └── frame_sampler.py   # Space-aware frame sampling
│   ├── models/                # Model implementations
│   │   ├── dinov3/           # DINOv3 integration (2D encoder)
│   │   ├── vggt/             # VGGT integration (3D encoder)
│   │   ├── llm/              # Spatial-MLLM backbone
│   │   ├── heatmap/          # Heatmap generation pipeline
│   │   ├── mlp/              # MLP token transformation
│   │   ├── feature_fusion.py # Advanced feature fusion
│   │   └── spatial_mllm_enhanced.py # Complete integration
│   └── utils/                 # Utility functions
│       ├── config.py         # Configuration management
│       └── logger.py         # Logging utilities
├── scripts/                   # Execution scripts
│   ├── train.py              # Training pipeline
│   ├── inference.py          # Inference pipeline
│   ├── evaluate.py           # Evaluation pipeline
│   └── preprocess.py         # Data preprocessing
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🔧 Configuration

The system uses YAML configuration files for comprehensive control:

### Key Configuration Sections

```yaml
# Model Architecture
dinov3:
  model_name: "dinov3_vit_large"
  freeze_backbone: true

vggt:
  img_size: 518
  geometry_head: true

llm:
  model_name: "Diankun/Spatial-MLLM-subset-sft"
  torch_dtype: "bfloat16"

# Video Processing
video:
  total_frames: 32      # N_m frames
  keyframes: 16         # N_k selected keyframes
  frame_size: [224, 224]

# Space-aware Frame Sampling
frame_sampling:
  method: "spatial_novelty"
  geometry_weight: 0.7
  camera_pose_weight: 0.8

# Training
training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 100
  stages:
    - name: "pretraining"
      focus: "heatmap_generation"
      freeze_llm: true
    - name: "finetuning"
      focus: "spatial_reasoning"
      freeze_llm: false
```

## 🎮 Usage Examples

### Training

```bash
# Basic training
python main.py --config configs/default_config.yaml --mode train

# Training with custom data path
python main.py --config configs/default_config.yaml --mode train --data_path /path/to/training/data

# Resume from checkpoint
python main.py --config configs/default_config.yaml --mode train --resume /path/to/checkpoint.pth

# Debug mode
python main.py --config configs/default_config.yaml --mode train --debug
```

### Inference

```bash
# Basic inference
python main.py --config configs/default_config.yaml --mode inference --video_path video.mp4

# With custom instruction
python main.py --config configs/default_config.yaml --mode inference \
    --video_path video.mp4 \
    --instruction "Navigate to the kitchen and find the red cup"

# Save to specific directory
python main.py --config configs/default_config.yaml --mode inference \
    --video_path video.mp4 \
    --output_dir /path/to/save/results
```

### Evaluation

```bash
# Evaluate on VSI-Bench (default)
python main.py --config configs/default_config.yaml --mode eval

# Evaluate on specific benchmark
python main.py --config configs/default_config.yaml --mode eval --benchmark RLBench

# All supported benchmarks
python main.py --config configs/default_config.yaml --mode eval --benchmark RLBench
python main.py --config configs/default_config.yaml --mode eval --benchmark COLOSSEUM  
python main.py --config configs/default_config.yaml --mode eval --benchmark GemBench
python main.py --config configs/default_config.yaml --mode eval --benchmark VSI-Bench
```

### Data Preprocessing

```bash
# Preprocess video dataset
python main.py --config configs/default_config.yaml --mode preprocess \
    --input_dir /path/to/raw/videos \
    --output_dir /path/to/processed/data

# This will:
# - Extract frames from videos (N_m frames per video)  
# - Validate video files
# - Process annotations
# - Create train/val/test splits
# - Generate metadata
```

## 📊 Output Examples

### Inference Results

After running inference, you'll get:

```
outputs/inference/
├── video_name_heatmaps.png          # Combined heatmap visualization
├── video_name_heatmap_1.png         # Individual heatmaps
├── video_name_heatmap_2.png         
├── video_name_selected_indices.npy  # Selected keyframe indices
├── video_name_geometry_info.npz     # Extracted geometry information
└── video_name_summary.json          # Results summary
```

### Training Outputs

```
logs/
├── vln_project.log                  # Training logs
├── checkpoints/                     # Model checkpoints
│   ├── checkpoint_epoch_10.pth
│   └── model_last.pth
└── visualizations/                  # Training visualizations
```

### Evaluation Results

```
results/evaluation/
├── RLBench_results.json            # Benchmark results
├── COLOSSEUM_results.json
├── GemBench_results.json
└── VSI-Bench_results.json
```

## 🔬 Core Algorithm: Space-Aware Frame Sampling

The heart of our system is the **Greedy Maximum Coverage Sampling Algorithm**:

```python
# Pseudo-code for space-aware sampling
def sample_keyframes(geometry_info, frame_indices):
    # 1. Extract spatial features from VGGT
    voxel_sets = discretize_spatial_coverage(geometry_info)
    
    # 2. Greedy maximum coverage selection
    S, C, R = [], set(), set(frame_indices)
    for t in range(target_frames):
        best_frame = argmax(coverage_gain(frame, C) for frame in R)
        S.append(best_frame)
        C.update(voxel_sets[best_frame])
        R.remove(best_frame)
    
    return S  # Selected N_k keyframes
```

This ensures we select the most spatially informative frames for processing.

## 🎯 First-Person Inter-Frame Heatmaps

Our system generates unique **first-person inter-frame heatmaps** that show:

1. **Current View**: What the model sees from the current frame
2. **Spatial Projections**: Where content from OTHER frames would appear in the current view
3. **Cross-Frame Understanding**: Spatial relationships between different temporal viewpoints
4. **3D Mental Model**: Evidence that the model builds coherent 3D scene understanding

### Example Scenario

```
Frame 1: Looking straight ahead at a table
Frame 5: Turned left, now seeing a chair  
Frame 10: Turned right, now seeing a window

→ Heatmap for Frame 1 shows:
  - High activation on the table (directly visible)
  - Medium activation to the left (where chair would appear)
  - Medium activation to the right (where window would appear)
```

## 🏗️ Advanced Features

### Multi-Stage Training

1. **Pretraining Phase**: Focus on heatmap generation with frozen LLM
2. **Fine-tuning Phase**: End-to-end spatial reasoning with unfrozen LLM

### Distributed Training

```bash
# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py --config configs/default_config.yaml --mode train
```

### Memory Optimization

```yaml
system:
  mixed_precision: true
  gradient_checkpointing: true  
  memory_efficient: true
  max_memory_gb: 24
```

## 📈 Performance Metrics

The system tracks comprehensive metrics:

- **Success Rate**: Task completion accuracy
- **Spatial Accuracy**: Heatmap-to-ground-truth alignment
- **Temporal Consistency**: Consistency across video frames
- **Inter-Frame Accuracy**: Cross-frame spatial understanding quality
- **Processing Efficiency**: Frames per second, keyframe selection quality
- **Heatmap Quality**: Peak clarity, dynamic range, spatial coherence

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   training:
     batch_size: 2  # Reduce from 4
   
   # Enable memory optimizations
   system:
     mixed_precision: true
     gradient_checkpointing: true
     memory_efficient: true
   ```

2. **Video Loading Errors**
   ```bash
   # Install opencv-python
   pip install opencv-python
   
   # Check video format compatibility
   # Supported: .mp4, .avi, .mov, .mkv, .webm
   ```

3. **Model Loading Issues**
   ```bash
   # Ensure transformers version compatibility
   pip install transformers==4.51.3
   
   # For flash attention issues:
   pip install flash-attn --no-build-isolation
   ```

### Debug Mode

```bash
python main.py --config configs/default_config.yaml --mode train --debug
```

This enables:
- Verbose logging
- Intermediate feature saving
- Memory and time profiling
- NaN/Inf value checking
- Sampling decision visualization

## 🔄 Development Workflow

### For Developers

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd VLN/Project
   conda create -n spatial-mllm python=3.10 -y
   conda activate spatial-mllm
   pip install -r requirements.txt
   ```

2. **Run Tests**
   ```bash
   # Test individual components
   python -m src.models.dinov3.example_usage
   python -m src.models.heatmap.demo
   python -m src.models.spatial_mllm_integration_example
   ```

3. **Development Commands**
   ```bash
   # Test with small dataset
   python main.py --config configs/default_config.yaml --mode train --debug
   
   # Quick inference test
   python main.py --config configs/default_config.yaml --mode inference \
     --video_path sample_video.mp4
   ```

## 📚 References and Architecture

This project builds upon several key research directions:

- **BridgeVLA**: 3D VLA framework and heatmap generation
- **Spatial-MLLM**: Video processing and LLM backbone architecture
- **DINOv3**: Self-supervised vision transformer for semantic understanding
- **VGGT**: Visual Geometry and Geometry Transformer for 3D understanding
- **Space-aware Sampling**: Novel contribution for efficient temporal processing

### Key Innovation

The **first-person inter-frame heatmap generation** is our core contribution, enabling models to:
- Understand spatial relationships across different temporal viewpoints
- Build coherent 3D mental models from video sequences  
- Generate actionable spatial attention maps for navigation
- Demonstrate cross-frame spatial reasoning capabilities

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the Apache License 2.0.

---

**🎯 Goal Achieved**: This system successfully generates first-person-view heatmaps that display spatial relationships between video frames, demonstrating advanced inter-frame spatial understanding for vision-language navigation tasks.