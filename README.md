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
# Single video processing (basic)
python main.py --video /path/to/video.mp4

# Single video with custom instruction
python main.py --video /path/to/video.mp4 --instruction "Navigate to the kitchen"

# Batch processing multiple videos
python main.py --batch video1.mp4 video2.mp4 video3.mp4

# Algorithm benchmarking
python main.py --benchmark --video /path/to/video.mp4

# Using configuration file
python main.py --config configs/custom.yaml --video /path/to/video.mp4
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

## 📋 Command Line Parameters

The `main.py` script provides a comprehensive command-line interface for all VLN operations:

### Core Input Options

| Parameter | Type | Description | Examples |
|-----------|------|-------------|----------|
| `--video` | str | Single video file path | `--video /path/to/video.mp4` |
| `--images` | str | Image sequence directory | `--images /path/to/frames/` |
| `--batch` | list | Multiple video files for batch processing | `--batch video1.mp4 video2.mp4 video3.mp4` |

### Processing Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--instruction` | str | "Navigate and analyze spatial relationships" | VLN navigation instruction text |
| `--algorithm` | str | "enhanced" | Sampling algorithm: `fast`, `quality`, `enhanced` |
| `--keyframes` | int | 8 | Number of keyframes to select from video |
| `--max_frames` | int | 32 | Maximum frames to load from video (candidate pool) |
| `--sample_fps` | float | None | Sample frames based on FPS (overrides max_frames) |

### Configuration and Output

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--config` | str | None | YAML configuration file path |
| `--output_dir` | str | "./outputs" | Output directory for results |
| `--no_visualization` | flag | False | Disable visualization output |
| `--no_save` | flag | False | Disable file saving |

### Operation Modes

| Parameter | Type | Description |
|-----------|------|-------------|
| `--benchmark` | flag | Run algorithm benchmark mode |
| `--profile` | flag | Enable performance profiling |
| `--verbose` / `-v` | flag | Enable verbose logging |
| `--optimize_memory` | flag | Optimize LLM memory usage |

### 🎮 Usage Examples

#### Single Video Processing
```bash
# Basic processing with default settings
python main.py --video test_video.mp4

# Custom instruction with specific algorithm
python main.py --video test_video.mp4 \
    --instruction "Navigate to the kitchen and find objects" \
    --algorithm enhanced \
    --keyframes 6

# High-resolution processing with more frames
python main.py --video test_video.mp4 \
    --max_frames 64 \
    --keyframes 16 \
    --verbose

# FPS-based sampling
python main.py --video test_video.mp4 \
    --sample_fps 2.0 \
    --keyframes 8
```

#### Batch Processing
```bash
# Process multiple videos with same settings
python main.py --batch \
    kitchen_video.mp4 \
    bedroom_video.mp4 \
    living_room.mp4 \
    --instruction "Analyze spatial layout" \
    --algorithm quality

# Batch processing with custom output directory
python main.py --batch *.mp4 \
    --output_dir ./batch_results \
    --keyframes 10
```

#### Algorithm Benchmarking
```bash
# Compare all algorithms on single video
python main.py --benchmark --video test_video.mp4

# Benchmark with performance profiling
python main.py --benchmark --video test_video.mp4 \
    --profile --verbose
```

#### Advanced Configuration
```bash
# Use custom configuration file
python main.py --config configs/high_quality.yaml \
    --video test_video.mp4

# Memory optimization for large videos
python main.py --video large_video.mp4 \
    --optimize_memory \
    --max_frames 24 \
    --keyframes 6

# Research mode with all outputs
python main.py --video research_video.mp4 \
    --verbose \
    --profile \
    --output_dir ./research_results
```

#### Image Sequence Processing
```bash
# Process image sequence (frames extracted from video)
python main.py --images /path/to/frame_sequence/ \
    --instruction "Navigate through the space"
```

### 🔍 Parameter Details

#### Algorithm Types
- **`fast`**: Quick uniform sampling, minimal computation
- **`quality`**: Balanced quality-speed tradeoff with spatial analysis
- **`enhanced`**: Advanced multi-objective submodular optimization

#### Frame Sampling Strategy
- **`max_frames`**: Sets candidate pool size (N_m frames loaded from video)
- **`keyframes`**: Final selection count (N_k keyframes processed by pipeline)
- **`sample_fps`**: Alternative sampling based on target framerate

#### Memory Optimization
- **`optimize_memory`**: Reduces LLM video pixel budget for large videos
- Automatically calculates optimal memory allocation based on frame count

#### Output Control
- **`no_visualization`**: Skips heatmap overlay generation (faster processing)
- **`no_save`**: Runs in preview mode without saving files
- **`verbose`**: Detailed progress logging and debugging information

### 💡 Pro Tips

1. **For Large Videos**: Use `--optimize_memory --max_frames 24 --keyframes 6`
2. **For Quality Results**: Use `--algorithm enhanced --keyframes 12 --max_frames 48`
3. **For Speed**: Use `--algorithm fast --keyframes 4 --no_visualization`
4. **For Debugging**: Add `--verbose --profile` to any command
5. **For Batch Jobs**: Use `--no_visualization` to speed up processing

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

## 🚀 Production-Ready VLN Pipeline

The main.py script provides a **production-ready implementation** for real-world VLN applications:

### 🎯 Key Features

- **Real-time inference**: Process videos and generate frame-indexed heatmaps
- **Multi-algorithm support**: Fast, quality, and enhanced sampling algorithms
- **Batch processing**: Handle multiple videos efficiently
- **Memory optimization**: Handle large videos with smart memory management
- **Comprehensive output**: Heatmaps, metrics, visualizations, and detailed logs
- **Flexible input**: Support for videos, image sequences, and batch processing

### 📊 Performance Characteristics

| Mode | Speed | Quality | Memory Usage | Best For |
|------|-------|---------|--------------|----------|
| `fast` | ⚡⚡⚡ | ⭐⭐ | 🟢 Low | Quick prototyping, batch jobs |
| `quality` | ⚡⚡ | ⭐⭐⭐ | 🟡 Medium | Balanced processing |
| `enhanced` | ⚡ | ⭐⭐⭐⭐ | 🔴 High | Research, high-quality results |

### 🔄 Typical Workflows

#### Research Workflow
```bash
# 1. Single video analysis with full output
python main.py --video research_sample.mp4 \
    --algorithm enhanced \
    --keyframes 12 \
    --max_frames 48 \
    --verbose \
    --profile

# 2. Algorithm comparison
python main.py --benchmark --video research_sample.mp4

# 3. Parameter exploration
python main.py --video research_sample.mp4 --keyframes 4 --algorithm fast
python main.py --video research_sample.mp4 --keyframes 8 --algorithm quality
python main.py --video research_sample.mp4 --keyframes 16 --algorithm enhanced
```

#### Production Workflow
```bash
# 1. Batch processing for deployment
python main.py --batch *.mp4 \
    --algorithm quality \
    --keyframes 8 \
    --output_dir ./production_results \
    --optimize_memory

# 2. High-throughput processing
python main.py --batch video_dataset/*.mp4 \
    --algorithm fast \
    --keyframes 6 \
    --no_visualization \
    --max_frames 24
```

#### Development Workflow
```bash
# 1. Quick testing
python main.py --video test_sample.mp4 --verbose

# 2. Memory-constrained development
python main.py --video large_test.mp4 \
    --optimize_memory \
    --max_frames 16 \
    --keyframes 4

# 3. Custom configuration testing
python main.py --config configs/dev_config.yaml \
    --video test_sample.mp4 \
    --verbose
```

## 📊 Output Examples

### Single Video Processing Results

After running `python main.py --video sample.mp4`, you'll get:

```
outputs/
├── video_sample_frame_5_heatmap_0.png   # Individual frame heatmaps
├── video_sample_frame_17_heatmap_1.png  # with overlay visualizations
├── video_sample_frame_23_heatmap_2.png
├── video_sample_frame_34_heatmap_3.png
├── summary_heatmaps.png                 # Combined visualization
├── metrics_sample.json                  # Processing metrics and results
└── algorithm_benchmark.json             # (if --benchmark used)
```

### Batch Processing Results

After running `python main.py --batch *.mp4 --output_dir batch_results`:

```
batch_results/
├── video_kitchen_frame_12_heatmap_0.png
├── video_kitchen_frame_28_heatmap_1.png
├── video_bedroom_frame_8_heatmap_0.png
├── video_bedroom_frame_15_heatmap_1.png
├── summary_heatmaps.png
├── metrics_kitchen.json
├── metrics_bedroom.json
└── batch_processing_summary.json
```

### Benchmark Results

After running `python main.py --benchmark --video test.mp4`:

```
outputs/
├── algorithm_benchmark.json         # Algorithm comparison results
├── fast_algorithm_heatmaps.png     # Results for each algorithm
├── quality_algorithm_heatmaps.png
├── enhanced_algorithm_heatmaps.png
└── benchmark_summary.json          # Performance comparison
```

### Sample Metrics Output

The `metrics_*.json` files contain detailed processing information:

```json
{
  "video_path": "/path/to/video.mp4",
  "instruction": "Navigate and analyze spatial relationships",
  "algorithm_type": "enhanced",
  "num_frames": 32,
  "num_keyframes": 8,
  "keyframe_indices": [5, 17, 23, 34, 51, 78, 89, 95],
  "heatmaps_shape": [8, 224, 224],
  "processing_time": {
    "video_loading": 1.23,
    "pipeline_processing": 28.45,
    "total": 29.68
  },
  "saved_heatmaps": [
    "outputs/video_sample_frame_5_heatmap_0.png",
    "outputs/video_sample_frame_17_heatmap_1.png"
  ],
  "summary_visualization": "outputs/summary_heatmaps.png"
}
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