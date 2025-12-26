# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .vision_transformer import (
    DinoVisionTransformer,
    vit_small,
    vit_base,
    vit_large,
    vit_giant,
)
from .aggregator import Aggregator
from .hub import (
    dinov3_vits16,
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vitg16,
    dinov3_vit7b16,
    load_dinov3_model,
    load_local_dinov3,
    DINOV3_MODELS,
)
from . import vggt_compat
from .compatibility import (
    DINOv3CompatibilityLayer,
    DINOv3CompatConfig,
    create_dinov3_compatibility_layer,
    DINOV3_EMBED_DIMS,
    validate_dinov3_config,
)

__all__ = [
    'DinoVisionTransformer',
    'vit_small',
    'vit_base',
    'vit_large',
    'vit_giant',
    'Aggregator',
    'dinov3_vits16',
    'dinov3_vitb16',
    'dinov3_vitl16',
    'dinov3_vitg16',
    'dinov3_vit7b16',
    'load_dinov3_model',
    'load_local_dinov3',
    'DINOV3_MODELS',
    'vggt_compat',
    # Compatibility layer
    'DINOv3CompatibilityLayer',
    'DINOv3CompatConfig',
    'create_dinov3_compatibility_layer',
    'DINOV3_EMBED_DIMS',
    'validate_dinov3_config',  # Config validation utility
]