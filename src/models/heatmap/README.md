# BridgeVLA-style Heatmap Module

This module provides comprehensive heatmap generation and visualization utilities adapted from BridgeVLA's approach to unified vision-language-action representation through 2D spatial heatmaps.

## Overview

BridgeVLA's key innovation is converting Large Language Model (LLM) token outputs into 2D spatial heatmaps that represent action predictions, object locations, and attention patterns. This enables unified vision-language-action learning where all modalities share the same 2D spatial representation space.

## Core Components

### 1. ConvexUpSample (`upsampling.py`)
Learned convex upsampling module that converts low-resolution feature maps (16x16) to high-resolution heatmaps (224x224).

**Key Features:**
- Convex combination weights ensure stable gradients
- Configurable upsampling ratios and kernel sizes  
- Optional batch normalization support
- Smooth interpolation with learned masks

### 2. LLMToHeatmapConverter (`converter.py`)
Complete pipeline for converting LLM hidden states to 2D heatmaps.

**Process:**
1. Extract vision tokens from LLM output
2. Reconstruct 1D token sequences into 2D spatial features
3. Apply ConvexUpSample to generate high-resolution heatmaps
4. Support multi-view camera inputs

### 3. Heatmap Generators (`generator.py`)
Utilities for creating target heatmaps from various annotation types.

**Supported Annotations:**
- Point coordinates → Gaussian heatmaps
- Bounding boxes → Center point heatmaps  
- Multi-point annotations → Fused heatmaps
- Grasp points for robotic manipulation

### 4. Visualization Tools (`visualizer.py`)
Comprehensive plotting and analysis utilities.

**Visualization Types:**
- Point annotations with heatmap overlays
- Bounding box visualizations
- Multi-view camera heatmaps
- Prediction vs target comparisons
- Animation frame generation

## Installation

The module is designed to work within the Project directory structure:

```
Project/
└── src/
    └── models/
        └── heatmap/
            ├── __init__.py
            ├── upsampling.py
            ├── converter.py
            ├── generator.py
            ├── visualizer.py
            ├── demo.py
            └── README.md
```

**Dependencies:**
- PyTorch (≥1.8.0)
- NumPy
- Matplotlib  
- PIL (Pillow)
- einops (for tensor reshaping)

## Quick Start

### Basic Usage

```python
from src.models.heatmap import (
    LLMToHeatmapConverter,
    generate_hm_from_pt,
    visualize_points_and_heatmap
)

# 1. Convert LLM outputs to heatmaps
converter = LLMToHeatmapConverter(vlm_dim=2048, patch_size=16, target_size=224)
heatmaps = converter(hidden_states, attention_mask, num_views=3)

# 2. Generate target heatmaps from points
points = torch.tensor([[112.0, 112.0], [56.0, 168.0]])  # Pixel coordinates
target_heatmap = generate_hm_from_pt(points, res=224, sigma=10.0)

# 3. Visualize results
from PIL import Image
image = Image.open('input_image.jpg')
points_norm = [(0.5, 0.5), (0.25, 0.75)]  # Normalized coordinates
visualize_points_and_heatmap(image, points_norm, heatmaps[0], 'output.png')
```

### Convenience Function

```python
from src.models.heatmap import quick_heatmap_from_llm

# Quick heatmap generation
heatmaps = quick_heatmap_from_llm(
    hidden_states, 
    attention_mask, 
    num_views=3,
    vlm_dim=2048
)
```

## Detailed Usage

### 1. LLM Token to Heatmap Conversion

```python
# Initialize converter
converter = LLMToHeatmapConverter(
    vlm_dim=2048,        # VLM hidden dimension
    patch_size=16,       # Spatial patch size  
    target_size=224,     # Output heatmap size
    up_kernel=3,         # Upsampling kernel size
    mask_scale=0.1       # Upsampling mask scale
)

# Convert LLM outputs
batch_size, seq_len, hidden_dim = 2, 300, 2048
hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
attention_mask = torch.ones(batch_size, seq_len)

heatmaps = converter(hidden_states, attention_mask, num_views=3)
print(f"Output shape: {heatmaps.shape}")  # [2, 3, 224, 224]
```

### 2. Target Heatmap Generation

```python
from src.models.heatmap import generate_target_heatmap_from_annotation

# Different annotation types
annotations = [
    ("[0.3, 0.4, 0.7, 0.6]", "detection_1"),      # Bounding box
    ("[[0.2, 0.3], [0.8, 0.7]]", "detection_2"),  # Multiple points
    ("[0.5, 0.5]", "single_point"),               # Single point
    ("[[0.3, 0.4], [0.6, 0.5]]", "grasp_point")   # Grasp candidates
]

for annotation, flag in annotations:
    target_hm = generate_target_heatmap_from_annotation(
        annotation, flag, image_size=224, sigma=8.0
    )
    print(f"{flag}: {target_hm.shape}")
```

### 3. Multi-Scale Heatmaps

```python
from src.models.heatmap import create_multi_scale_heatmap

points = torch.tensor([[112.0, 112.0]])
multi_scale_hm = create_multi_scale_heatmap(
    points, 
    image_size=224,
    scales=[2.0, 5.0, 10.0]  # Different Gaussian sigmas
)
```

### 4. Advanced Visualization

```python
from src.models.heatmap import (
    visualize_multi_view_heatmaps,
    visualize_attention_comparison
)

# Multi-view visualization
images = [img1, img2, img3]  # PIL Images
heatmaps = torch.randn(3, 224, 224)  # 3 views
visualize_multi_view_heatmaps(
    images, heatmaps, 'multiview.png',
    view_names=['Front', 'Side', 'Top']
)

# Prediction vs target comparison
visualize_attention_comparison(
    image, predicted_heatmap, target_heatmap, 'comparison.png'
)
```

## Demo Script

Run the comprehensive demo to see all functionality:

```bash
# Run all demos
python src/models/heatmap/demo.py --mode all --output_dir ./heatmap_demos

# Run specific demos
python src/models/heatmap/demo.py --mode basic
python src/models/heatmap/demo.py --mode llm_to_heatmap  
python src/models/heatmap/demo.py --mode training
python src/models/heatmap/demo.py --mode upsampling
python src/models/heatmap/demo.py --mode visualization
```

**Demo Outputs:**
- `demo_basic_heatmaps.png` - Gaussian heatmap generation
- `demo_llm_to_heatmap.png` - LLM token conversion
- `demo_training_pipeline.png` - Training target generation
- `demo_convex_upsampling.png` - Upsampling examples
- `demo_point_visualization.png` - Point annotation visualization
- `demo_bbox_visualization.png` - Bounding box visualization

## Technical Details

### ConvexUpSample Architecture

The ConvexUpSample module uses learned convex combinations for upsampling:

1. **Feature Generation Network**: Processes input features through conv layers
2. **Mask Generation Network**: Creates upsampling weight masks
3. **Convex Combination**: Uses softmax-normalized masks for stable upsampling
4. **Unfold Operation**: Efficiently applies convex weights across spatial regions

### LLM Token Processing

The token-to-heatmap conversion follows these steps:

1. **Token Extraction**: Extract first N image tokens from LLM hidden states
2. **Spatial Reconstruction**: Reshape 1D tokens to 2D spatial layout (16x16)  
3. **Multi-View Handling**: Support multiple camera views simultaneously
4. **Upsampling**: Generate high-resolution heatmaps via ConvexUpSample

### Training Integration

For training integration with your model:

```python
import torch.nn.functional as F
from src.models.heatmap import generate_target_heatmap_from_annotation

# In your training loop
def compute_heatmap_loss(pred_heatmaps, annotations, flags):
    losses = []
    for pred, annotation, flag in zip(pred_heatmaps, annotations, flags):
        target = generate_target_heatmap_from_annotation(annotation, flag)
        target = target.to(pred.device)
        
        # Flatten for cross-entropy loss
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        loss = F.cross_entropy(pred_flat.unsqueeze(0), target_flat.unsqueeze(0))
        losses.append(loss)
    
    return torch.stack(losses).mean()
```

## Applications

### Robotic Manipulation
- **Grasp Point Prediction**: Convert "pick up the cup" → grasp location heatmap
- **Placement Planning**: Convert "place on the table" → placement heatmap
- **Multi-Step Actions**: Sequence of heatmaps for complex tasks

### Object Detection  
- **Visual Grounding**: Convert "find the red apple" → detection heatmap
- **Referring Expression**: Natural language → spatial attention
- **Multi-Object Tracking**: Track objects through heatmap sequences

### Scene Understanding
- **Attention Visualization**: Show what the model focuses on
- **Spatial Reasoning**: "What's to the left of X?" → spatial heatmap
- **Navigation Planning**: Path planning through spatial predictions

## Performance Considerations

### Memory Usage
- ConvexUpSample uses ~8MB additional memory per batch
- Multi-view processing scales linearly with number of views
- Consider using gradient checkpointing for large batch sizes

### Speed Optimization
- Use `torch.jit.script` for ConvexUpSample in production
- Cache converter instances to avoid repeated initialization
- Use half-precision (fp16) for inference if memory-constrained

### Quality Tips
- Sigma values 5-15 work well for most applications
- Use multi-scale heatmaps for better coverage
- Apply augmentation during training for robustness

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Ensure LLM outputs have sufficient image tokens (≥256 per view)
2. **CUDA Errors**: Check device consistency between tensors
3. **Visualization Issues**: Verify PIL image formats (RGB vs RGBA)
4. **Memory Errors**: Reduce batch size or use gradient checkpointing

### Debug Tips

```python
# Enable debug mode
converter = LLMToHeatmapConverter(vlm_dim=2048)
heatmaps, features = converter.forward_with_features(hidden_states, attention_mask)
print(f"Features shape: {features.shape}")  # Check intermediate outputs

# Visualize upsampling process
upsampler = ConvexUpSample(in_dim=2048, out_dim=1, up_ratio=14)
x = torch.randn(1, 2048, 16, 16)
y = upsampler(x)
print(f"Upsampling: {x.shape} -> {y.shape}")
```

## References

1. **BridgeVLA**: [Original implementation](https://github.com/BridgeVLA/BridgeVLA)
2. **RVT**: ConvexUpSample adapted from [RVT repository](https://github.com/NVlabs/RVT)
3. **PaliGemma**: Base VLM architecture from Google Research

## Contributing

This module is adapted from BridgeVLA for the Project directory. When making modifications:

1. Maintain compatibility with existing BridgeVLA patterns
2. Add comprehensive tests for new functionality
3. Update documentation and examples
4. Follow the existing code style and structure

## License

This module adapts code from BridgeVLA and RVT, which are subject to their respective licenses. Please refer to the original repositories for license details.