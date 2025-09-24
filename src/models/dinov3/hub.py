# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.hub
from typing import Optional, Union
from pathlib import Path
import json
from safetensors import safe_open
from safetensors.torch import load_file

from .vision_transformer import vit_small, vit_base, vit_large, vit_giant, DinoVisionTransformer


def dinov3_vits16(
    pretrained: bool = True,
    weights: Optional[str] = None,
    **kwargs
):
    """
    DINOv3 ViT-Small model with 16x16 patches.

    Args:
        pretrained (bool): If True, load pretrained weights
        weights (str, optional): Path to custom weights file
        **kwargs: Additional arguments passed to the model

    Returns:
        DinoVisionTransformer: The model instance
    """
    model_args = {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4,
        "qkv_bias": True,
        "layerscale_init": 1e-5,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "n_storage_tokens": 4,
        "pos_embed_rope_base": 100.0,
        "pos_embed_rope_rescale_coords": 2.0,
        **kwargs
    }

    model = DinoVisionTransformer(**model_args)

    if pretrained and weights:
        if Path(weights).is_dir() or (Path(weights).exists() and weights.endswith('.safetensors')):
            model = load_safetensors_weights(model, weights)
        else:
            print(f"Warning: Weights path {weights} not found or not safetensors format")
    elif not pretrained:
        model.init_weights()

    return model


def dinov3_vitb16(
    pretrained: bool = True,
    weights: Optional[str] = None,
    **kwargs
):
    """
    DINOv3 ViT-Base model with 16x16 patches.

    Args:
        pretrained (bool): If True, load pretrained weights
        weights (str, optional): Path to custom weights file
        **kwargs: Additional arguments passed to the model

    Returns:
        DinoVisionTransformer: The model instance
    """
    model_args = {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_ratio": 4,
        "qkv_bias": True,
        "layerscale_init": 1e-5,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "n_storage_tokens": 4,
        "pos_embed_rope_base": 100.0,
        "pos_embed_rope_rescale_coords": 2.0,
        **kwargs
    }

    model = DinoVisionTransformer(**model_args)

    if pretrained and weights:
        if Path(weights).is_dir() or (Path(weights).exists() and weights.endswith('.safetensors')):
            model = load_safetensors_weights(model, weights)
        else:
            print(f"Warning: Weights path {weights} not found or not safetensors format")
    elif not pretrained:
        model.init_weights()

    return model


def dinov3_vitl16(
    pretrained: bool = True,
    weights: Optional[str] = None,
    **kwargs
):
    """
    DINOv3 ViT-Large model with 16x16 patches.

    Args:
        pretrained (bool): If True, load pretrained weights
        weights (str, optional): Path to custom weights file
        **kwargs: Additional arguments passed to the model

    Returns:
        DinoVisionTransformer: The model instance
    """
    model_args = {
        "patch_size": 16,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_ratio": 4,
        "qkv_bias": True,
        "layerscale_init": 1e-5,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "n_storage_tokens": 4,
        "pos_embed_rope_base": 100.0,
        "pos_embed_rope_rescale_coords": 2.0,
        **kwargs
    }

    model = DinoVisionTransformer(**model_args)

    if pretrained and weights:
        if Path(weights).is_dir() or (Path(weights).exists() and weights.endswith('.safetensors')):
            model = load_safetensors_weights(model, weights)
        else:
            print(f"Warning: Weights path {weights} not found or not safetensors format")
    elif not pretrained:
        model.init_weights()

    return model


def dinov3_vitg16(
    pretrained: bool = True,
    weights: Optional[str] = None,
    **kwargs
):
    """
    DINOv3 ViT-Giant model with 16x16 patches.

    Args:
        pretrained (bool): If True, load pretrained weights
        weights (str, optional): Path to custom weights file
        **kwargs: Additional arguments passed to the model

    Returns:
        DinoVisionTransformer: The model instance
    """
    model_args = {
        "patch_size": 16,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "ffn_ratio": 4,
        "qkv_bias": True,
        "layerscale_init": 1e-5,
        "norm_layer": "layernormbf16",
        "ffn_layer": "mlp",
        "n_storage_tokens": 4,
        "pos_embed_rope_base": 100.0,
        "pos_embed_rope_rescale_coords": 2.0,
        **kwargs
    }

    model = DinoVisionTransformer(**model_args)

    if pretrained and weights:
        if Path(weights).is_dir() or (Path(weights).exists() and weights.endswith('.safetensors')):
            model = load_safetensors_weights(model, weights)
        else:
            print(f"Warning: Weights path {weights} not found or not safetensors format")
    elif not pretrained:
        model.init_weights()

    return model


def dinov3_vit7b16(
    pretrained: bool = True,
    weights: Optional[str] = None,
    **kwargs
):
    """
    DINOv3 ViT-7B model with 16x16 patches (matches your downloaded model).

    Args:
        pretrained (bool): If True, load pretrained weights
        weights (str, optional): Path to custom weights file
        **kwargs: Additional arguments passed to the model

    Returns:
        DinoVisionTransformer: The model instance
    """
    model_args = {
        "patch_size": 16,
        "embed_dim": 4096,
        "depth": 40,
        "num_heads": 32,
        "ffn_ratio": 2.0,  # 8192 / 4096 = 2.0 based on config
        "qkv_bias": False,  # Based on config
        "layerscale_init": 1.0,  # Based on config
        "norm_layer": "layernormbf16",
        "ffn_layer": "swiglu" if kwargs.get('use_gated_mlp', True) else "mlp",
        "n_storage_tokens": 4,
        "pos_embed_rope_base": 100.0,
        "pos_embed_rope_rescale_coords": 2.0,
        **kwargs
    }

    model = DinoVisionTransformer(**model_args)

    if pretrained and weights:
        if Path(weights).is_dir() or (Path(weights).exists() and weights.endswith('.safetensors')):
            model = load_safetensors_weights(model, weights)
        else:
            print(f"Warning: Weights path {weights} not found or not safetensors format")
    elif not pretrained:
        model.init_weights()

    return model


# Model registry for easy access
DINOV3_MODELS = {
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vitg16": dinov3_vitg16,
    "dinov3_vit7b16": dinov3_vit7b16,
}

# Convenience function for your downloaded model
def load_local_dinov3(model_path="./models/dinov3"):
    """
    Load the DINOv3 model from your local safetensors files.

    Args:
        model_path: Path to directory containing config.json and safetensors files

    Returns:
        Loaded DINOv3 model
    """
    return load_dinov3_from_config(model_path)


def load_dinov3_model(model_name: str, **kwargs):
    """
    Load a DINOv3 model by name.
    
    Args:
        model_name (str): Name of the model (e.g., "dinov3_vitl16")
        **kwargs: Additional arguments passed to the model constructor
    
    Returns:
        DinoVisionTransformer: The model instance
    """
    if model_name not in DINOV3_MODELS:
        available = list(DINOV3_MODELS.keys())
        raise ValueError(f"Model {model_name} not available. Available models: {available}")
    
    return DINOV3_MODELS[model_name](**kwargs)


def load_safetensors_weights(model, safetensors_path: str):
    """
    Load weights from safetensors format with proper key mapping.

    Args:
        model: The model instance to load weights into
        safetensors_path: Path to safetensors file or directory containing sharded files

    Returns:
        The model with loaded weights
    """
    safetensors_path = Path(safetensors_path)

    try:
        state_dict = {}

        if safetensors_path.is_dir():
            # Load sharded safetensors
            index_file = safetensors_path / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)

                weight_map = index["weight_map"]
                for param_name, file_name in weight_map.items():
                    file_path = safetensors_path / file_name
                    if file_path.exists():
                        with safe_open(file_path, framework="pt", device="cpu") as f:
                            if param_name in f.keys():
                                state_dict[param_name] = f.get_tensor(param_name)
            else:
                # Load all safetensor files in directory
                for safetensor_file in safetensors_path.glob("*.safetensors"):
                    if "index" not in safetensor_file.name:
                        shard_state_dict = load_file(safetensor_file)
                        state_dict.update(shard_state_dict)
        else:
            # Single safetensors file
            state_dict = load_file(safetensors_path)

        print(f"Loaded {len(state_dict)} parameters from safetensors")

        # Map HuggingFace DINOv3 keys to our implementation
        model_state_dict = {}
        model_keys = set(model.state_dict().keys())

        key_mapping = {
            # Patch embedding
            "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
            "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",

            # CLS token and register tokens
            "embeddings.cls_token": "cls_token",
            "embeddings.register_tokens": "storage_tokens",

            # Mask token
            "embeddings.mask_token": "mask_token",

            # Layer norm
            "layernorm.weight": "norm.weight",
            "layernorm.bias": "norm.bias",
        }

        # Process transformer blocks
        for i in range(model.n_blocks):
            block_prefix = f"encoder.layer.{i}"
            target_prefix = f"blocks.{i}"

            # Attention layers
            key_mapping.update({
                f"{block_prefix}.attention.attention.query.weight": f"{target_prefix}.attn.qkv.weight",
                f"{block_prefix}.attention.attention.query.bias": f"{target_prefix}.attn.qkv.bias",
                f"{block_prefix}.attention.output.dense.weight": f"{target_prefix}.attn.proj.weight",
                f"{block_prefix}.attention.output.dense.bias": f"{target_prefix}.attn.proj.bias",

                # Layer norms
                f"{block_prefix}.layernorm_before.weight": f"{target_prefix}.norm1.weight",
                f"{block_prefix}.layernorm_before.bias": f"{target_prefix}.norm1.bias",
                f"{block_prefix}.layernorm_after.weight": f"{target_prefix}.norm2.weight",
                f"{block_prefix}.layernorm_after.bias": f"{target_prefix}.norm2.bias",

                # MLP layers
                f"{block_prefix}.mlp.fc1.weight": f"{target_prefix}.mlp.fc1.weight",
                f"{block_prefix}.mlp.fc1.bias": f"{target_prefix}.mlp.fc1.bias",
                f"{block_prefix}.mlp.fc2.weight": f"{target_prefix}.mlp.fc2.weight",
                f"{block_prefix}.mlp.fc2.bias": f"{target_prefix}.mlp.fc2.bias",

                # LayerScale parameters
                f"{block_prefix}.layer_scale1.lambda1": f"{target_prefix}.ls1.gamma",
                f"{block_prefix}.layer_scale2.lambda1": f"{target_prefix}.ls2.gamma",
            })

        # Apply key mapping and filter compatible weights
        mapped_state_dict = {}
        skipped_keys = []
        shape_mismatches = []

        for hf_key, weight in state_dict.items():
            if hf_key in key_mapping:
                our_key = key_mapping[hf_key]
                if our_key in model_keys:
                    model_weight_shape = model.state_dict()[our_key].shape
                    if weight.shape == model_weight_shape:
                        mapped_state_dict[our_key] = weight
                    else:
                        shape_mismatches.append((hf_key, our_key, weight.shape, model_weight_shape))
                else:
                    skipped_keys.append(f"{hf_key} -> {our_key} (not in model)")
            else:
                # Try direct key matching
                if hf_key in model_keys:
                    model_weight_shape = model.state_dict()[hf_key].shape
                    if weight.shape == model_weight_shape:
                        mapped_state_dict[hf_key] = weight
                    else:
                        shape_mismatches.append((hf_key, hf_key, weight.shape, model_weight_shape))
                else:
                    skipped_keys.append(f"{hf_key} (no mapping found)")

        # Load mapped weights
        missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False)

        print(f"Successfully loaded {len(mapped_state_dict)}/{len(model_keys)} parameters")

        if missing_keys:
            print(f"Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")

        if unexpected_keys:
            print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

        if shape_mismatches:
            print(f"Shape mismatches ({len(shape_mismatches)}):")
            for hf_key, our_key, hf_shape, our_shape in shape_mismatches[:3]:
                print(f"  {hf_key} -> {our_key}: HF {hf_shape} vs Ours {our_shape}")

        if skipped_keys:
            print(f"Skipped keys ({len(skipped_keys)}): {skipped_keys[:5]}{'...' if len(skipped_keys) > 5 else ''}")

        return model

    except Exception as e:
        print(f"Error loading safetensors weights: {e}")
        import traceback
        traceback.print_exc()
        return model


def get_dinov3_model_urls():
    """
    Get URLs for pretrained DINOv3 models.
    
    Returns:
        Dictionary mapping model names to download URLs
    """
    # These would be the actual DINOv3 model URLs
    # For now, return empty dict as we don't have access to official weights
    return {
        "dinov3_vits14": None,
        "dinov3_vitb14": None, 
        "dinov3_vitl14": None,
        "dinov3_vitg14": None,
    }


def load_dinov3_from_config(config_path: str, weights_path: Optional[str] = None):
    """
    Load DINOv3 model from HuggingFace config and safetensors weights.

    Args:
        config_path: Path to config.json file or directory containing it
        weights_path: Optional path to safetensors weights

    Returns:
        Loaded DINOv3 model
    """
    config_path = Path(config_path)
    if config_path.is_dir():
        config_file = config_path / "config.json"
    else:
        config_file = config_path
        config_path = config_file.parent

    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Map HuggingFace config to our model parameters
    model_args = {
        "img_size": config["image_size"],
        "patch_size": config["patch_size"],
        "in_chans": config["num_channels"],
        "embed_dim": config["hidden_size"],
        "depth": config["num_hidden_layers"],
        "num_heads": config["num_attention_heads"],
        "ffn_ratio": config["intermediate_size"] / config["hidden_size"],
        "qkv_bias": config.get("query_bias", True) or config.get("key_bias", True) or config.get("value_bias", True),
        "proj_bias": config.get("proj_bias", True),
        "ffn_bias": config.get("mlp_bias", True),
        "n_storage_tokens": config.get("num_register_tokens", 4),
        "layerscale_init": config.get("layerscale_value", 1e-5),
        "norm_layer": "layernormbf16",
        "ffn_layer": "swiglu" if config.get("use_gated_mlp", False) else "mlp",
        "pos_embed_rope_base": config.get("rope_theta", 100.0),
        "pos_embed_rope_rescale_coords": config.get("pos_embed_rescale", 2.0),
        "pos_embed_rope_shift_coords": config.get("pos_embed_shift"),
        "pos_embed_rope_jitter_coords": config.get("pos_embed_jitter"),
        "pos_embed_rope_dtype": "fp32",
    }

    print(f"Creating DINOv3 model with config: {model_args}")
    model = DinoVisionTransformer(**model_args)

    # Load weights if provided
    if weights_path:
        model = load_safetensors_weights(model, weights_path)
    elif (config_path / "model.safetensors").exists():
        model = load_safetensors_weights(model, config_path / "model.safetensors")
    elif (config_path / "model.safetensors.index.json").exists():
        model = load_safetensors_weights(model, config_path)
    else:
        print("No weights found, using random initialization")
        model.init_weights()

    return model


def verify_model_integrity(model, test_input_size=(1, 3, 224, 224)):
    """
    Verify that the model is working correctly.

    Args:
        model: The model to verify
        test_input_size: Size of test input tensor

    Returns:
        bool: True if model passes basic tests
    """
    try:
        model.eval()

        # Create test input
        test_input = torch.randn(test_input_size)

        # Test forward pass
        with torch.no_grad():
            output = model(test_input, is_training=False)

        # Verify output format
        if isinstance(output, dict):
            required_keys = ["x_norm_clstoken", "x_norm_patchtokens"]
            for key in required_keys:
                if key not in output:
                    print(f"Missing output key: {key}")
                    return False
        elif isinstance(output, torch.Tensor):
            # Direct output mode
            if len(output.shape) != 2:  # Should be [B, embed_dim]
                print(f"Unexpected output shape: {output.shape}")
                return False
        else:
            print(f"Unexpected output type: {type(output)}")
            return False

        print("Model integrity verification passed")
        return True

    except Exception as e:
        print(f"Model integrity verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False