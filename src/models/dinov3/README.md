# DINOv3 Integration for Your Project

This module provides a complete integration of Meta's DINOv3 (Self-Supervised Vision Transformers) into your project, following the architecture patterns established in Spatial-MLLM but adapted for DINOv3's API and capabilities.

## Overview

DINOv3 represents the next generation of self-supervised vision transformers, offering improved performance and capabilities over DINOv2. This integration provides:

- **Full API Compatibility**: Seamless integration with VGGT's interface
- **Multi-frame Processing**: Support for temporal sequences via aggregator
- **Flexible Architecture**: Multiple model sizes and configurations
- **Production Ready**: Optimized for both research and deployment

## Key Features

### 🔧 **VGGT API Compatibility**
- Full compatibility with VGGT's layer interface
- Seamless drop-in replacement for DINOv2 in existing code
- Maintains the same function signatures and behaviors

### 🚀 **Multiple Model Variants**
- **dinov3_vits16**: Small model (384 dim, 12 layers)
- **dinov3_vitb16**: Base model (768 dim, 12 layers)  
- **dinov3_vitl16**: Large model (1024 dim, 24 layers)
- **dinov3_vitg16**: Giant model (1536 dim, 40 layers)
- **dinov3_vitl14**: Large model with 14×14 patches

### 🎯 **Advanced Aggregator**
- Multi-frame temporal processing
- Alternating attention mechanisms (frame-level + global)
- RoPE position encoding support
- Configurable attention patterns

### 📦 **Hub Interface**
- Easy model loading with pretrained weights
- Automatic weight downloading and caching
- Custom weight loading support

## Quick Start

### Basic Usage

```python
from src.models.dinov3 import dinov3_vitl16

# Load a pretrained DINOv3 model
model = dinov3_vitl16(pretrained=True)

# Process images
import torch
images = torch.randn(2, 3, 224, 224)  # [batch, channels, height, width]
features = model.forward_features(images)

print(f"CLS token: {features['x_norm_clstoken'].shape}")      # [2, 1024]
print(f"Patch tokens: {features['x_norm_patchtokens'].shape}") # [2, 196, 1024]
```

### Multi-frame Processing with Aggregator

```python
from src.models.dinov3 import Aggregator

# Create aggregator for multi-frame processing
aggregator = Aggregator(
    img_size=224,
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    n_storage_tokens=4,
    patch_embed="dinov3_vitl16_reg",
    aa_order=["frame", "global"],
)

# Process video sequences
video_frames = torch.randn(2, 8, 3, 224, 224)  # [batch, frames, channels, height, width]
output_list, patch_start_idx = aggregator(video_frames)

print(f"Output features: {output_list[-1].shape}")  # [2, 8, 200, 2048]
```

### Using the Integrated Model

```python
from src.models.dinov3.example_usage import DINOv3IntegratedModel

# For single-frame processing
model = DINOv3IntegratedModel(
    dinov3_model="dinov3_vitb16",
    use_aggregator=False,
    pretrained=True
)

images = torch.randn(4, 3, 224, 224)
output = model(images)
print(f"CLS token: {output['cls_token'].shape}")

# For multi-frame processing  
model_multi = DINOv3IntegratedModel(
    dinov3_model="dinov3_vitl16", 
    use_aggregator=True,
    aa_order=["frame", "global"]
)

video = torch.randn(2, 6, 3, 224, 224)
output = model_multi(video)
print(f"Features: {output['features'].shape}")
```

## Architecture Details

### Vision Transformer Components

The DINOv3 integration includes all core components:

- **DinoVisionTransformer**: Main transformer architecture
- **PatchEmbed**: Image to patch embedding 
- **Block**: Transformer block with self-attention and FFN
- **Attention**: Multi-head self-attention with optional QK normalization
- **Mlp/SwiGLU**: Feed-forward network options

### VGGT Compatibility Layer

Located in `vggt_compat/`, this module provides:

- **layers.py**: All VGGT layer implementations adapted for DINOv3
- **rope.py**: 2D Rotary Position Embeddings for spatial data
- Full API compatibility with existing VGGT-based code

### Aggregator Architecture

The aggregator implements alternating attention over multiple frames:

1. **Frame Attention**: Process each frame independently
2. **Global Attention**: Process all frames together  
3. **Alternating Pattern**: Configurable attention order
4. **Position Encoding**: RoPE-based spatial position encoding

## Configuration

### YAML Configuration

See `configs/dinov3_config.yaml` for complete configuration options:

```yaml
model:
  backbone:
    dinov3_model: "dinov3_vitl14"
    img_size: 518
    patch_size: 14
    embed_dim: 1024
    depth: 24
    num_heads: 16
    n_storage_tokens: 4
    
  aggregator:
    enabled: true
    aa_order: ["frame", "global"]
    rope_freq: 100
```

### Model Variants

Pre-configured variants for different use cases:

- **Lightweight**: Fast inference with small models
- **High Performance**: Best accuracy with large models  
- **Video**: Optimized for temporal sequences

## Integration with Spatial-MLLM Pattern

This integration follows Spatial-MLLM's established patterns:

1. **Similar API**: Same function signatures and interfaces
2. **Aggregator Design**: Multi-frame processing capabilities
3. **Configuration**: YAML-based configuration system
4. **Hub Interface**: Easy model loading and management

### Key Differences from DINOv2

- **Storage Tokens**: DINOv3 uses storage tokens instead of register tokens
- **RoPE Integration**: Built-in rotary position encoding support
- **Enhanced Architecture**: Improved transformer blocks and attention
- **Better Performance**: Superior representation learning capabilities

## Testing

Run the test suite to verify installation:

```bash
python test_dinov3_integration.py
```

This will test:
- ✅ Basic model creation
- ✅ Hub model loading  
- ✅ Forward pass functionality
- ✅ Aggregator processing
- ✅ Integrated model usage
- ✅ Configuration loading

## File Structure

```
dinov3/
├── __init__.py              # Main module exports
├── vision_transformer.py   # Core DINOv3 transformer
├── aggregator.py           # Multi-frame aggregator  
├── hub.py                  # Model hub interface
├── example_usage.py        # Usage examples
├── vggt_compat/           # VGGT compatibility layer
│   ├── __init__.py
│   ├── layers.py          # Compatible layer implementations
│   └── rope.py            # 2D rotary position embeddings
└── README.md              # This file
```

## Requirements

Add to your `requirements.txt`:

```txt
torch>=2.0.0
torchvision
xformers  # Optional, for memory-efficient attention
fvcore    # For DINOv3 utilities
```

## Migration from DINOv2

To migrate from DINOv2 to DINOv3:

1. **Update imports**:
   ```python
   # Old
   from spatial_mllm.models.vggt import Aggregator
   
   # New  
   from src.models.dinov3 import Aggregator
   ```

2. **Update model names**:
   ```python
   # Old
   patch_embed="dinov2_vitl14_reg"
   
   # New
   patch_embed="dinov3_vitl14_reg"  
   ```

3. **Update token names**:
   ```python
   # Old: register tokens
   num_register_tokens=4
   
   # New: storage tokens
   n_storage_tokens=4
   ```

## Performance Notes

- **Memory**: DINOv3 models are larger than DINOv2, plan memory accordingly
- **Speed**: Similar inference speed with better accuracy
- **Scaling**: Giant model (vitg16) requires significant GPU memory
- **Optimization**: Use `xformers` for memory-efficient attention when available

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce batch size or use gradient checkpointing
3. **CUDA Errors**: Verify PyTorch CUDA compatibility

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger("dinov3").setLevel(logging.DEBUG)
```

## Contributing

To extend this integration:

1. Follow the established patterns in `vggt_compat/`
2. Maintain API compatibility with VGGT
3. Add tests for new functionality
4. Update configuration files as needed

## License

This integration follows the same license terms as the original DINOv3 implementation from Meta Platforms, Inc.