"""
Image augmentation transforms for VLN datasets.

All transforms operate on numpy uint8 arrays [H, W, 3] and preserve
geometric relationships (safe for spatial tasks).
"""

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class ColorJitterAugmentation:
    """Color jitter — brightness / contrast / saturation / hue."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: [H, W, 3] RGB image, uint8
        Returns:
            augmented image
        """
        if random.random() > self.p:
            return image

        image = image.astype(np.float32)

        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            image = image * factor

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean()
            image = (image - mean) * factor + mean

        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            pil_image = Image.fromarray(image.clip(0, 255).astype(np.uint8))
            pil_image = ImageEnhance.Color(pil_image).enhance(max(factor, 0.0))
            image = np.asarray(pil_image, dtype=np.float32)

        if self.hue > 0:
            shift = int(round(random.uniform(-self.hue, self.hue) * 255))
            pil_image = Image.fromarray(image.clip(0, 255).astype(np.uint8)).convert("HSV")
            hsv = np.asarray(pil_image, dtype=np.uint8).copy()
            hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + shift) % 256).astype(np.uint8)
            image = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"), dtype=np.float32)

        return np.clip(image, 0, 255).astype(np.uint8)


class GaussianNoiseAugmentation:
    """Add Gaussian noise to improve robustness."""

    def __init__(self, std: float = 10.0, p: float = 0.3):
        self.std = std
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        noise = np.random.normal(0, self.std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class InternNavStyleAugmentation:
    """InternNav-style augmentation (posterize / sharpness / autocontrast).

    Aligned with InternNav ``train_dual_system.sh`` augmentation pipeline,
    implemented with PIL.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image

        pil_img = Image.fromarray(image)

        if random.random() > 0.5:
            pil_img = ImageOps.posterize(pil_img, bits=4)

        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.5)

        if random.random() > 0.5:
            pil_img = ImageOps.autocontrast(pil_img)

        return np.asarray(pil_img)
