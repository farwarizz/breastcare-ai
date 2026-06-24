"""
src/ml/image_preprocessing.py — Mammogram image preprocessing pipeline.

Applies medical image preprocessing standard to mammography:
- CLAHE contrast enhancement for dense tissue visibility
- Breast region segmentation (background removal)
- Test-time augmentation (TTA) for robust inference

Usage:
    from src.ml.image_preprocessing import preprocess_mammogram, tta_predict
    processed = preprocess_mammogram(pil_image)
    avg_prob = tta_predict(model, transform, pil_image)
"""
from __future__ import annotations

import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage, ImageFilter
    import torch
    from torchvision import transforms as T
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


def preprocess_mammogram(pil_image: Any) -> Any:
    """
    Apply medical image preprocessing to a mammogram.

    Steps:
        1. Convert to grayscale (if RGB)
        2. CLAHE-like contrast enhancement
        3. Breast region detection via thresholding
        4. Normalize to 0-255 range
        5. Convert back to RGB (for model input)

    Returns a preprocessed PIL Image.
    """
    if not _DEPS_OK:
        raise ImportError("PIL and torch required: pip install Pillow torch torchvision")

    # Convert to grayscale
    img = pil_image.convert("L")
    arr = np.array(img, dtype=np.float32)

    # Simple CLAHE-like local contrast enhancement
    # (true CLAHE requires opencv, this is a portable approximation)
    arr = _adaptive_histogram_equalization(arr)

    # Breast region segmentation via Otsu-like thresholding
    threshold = np.percentile(arr[arr > 0], 5) if np.any(arr > 0) else 10
    mask = arr > threshold
    arr[~mask] = 0

    # Normalize to 0-255
    if arr.max() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255

    arr = arr.astype(np.uint8)
    processed = PILImage.fromarray(arr).convert("RGB")

    return processed


def _adaptive_histogram_equalization(
    arr: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8
) -> np.ndarray:
    """
    Approximate CLAHE (Contrast Limited Adaptive Histogram Equalization).

    This is a simplified version that doesn't require OpenCV.
    For production, use cv2.createCLAHE().
    """
    h, w = arr.shape
    result = np.zeros_like(arr)

    step_h = max(h // tile_size, 1)
    step_w = max(w // tile_size, 1)

    for i in range(0, h, step_h):
        for j in range(0, w, step_w):
            tile = arr[i:i + step_h, j:j + step_w]
            if tile.size == 0:
                continue

            # Local histogram equalization
            tile_min, tile_max = tile.min(), tile.max()
            if tile_max > tile_min:
                normalized = (tile - tile_min) / (tile_max - tile_min) * 255
                result[i:i + step_h, j:j + step_w] = normalized
            else:
                result[i:i + step_h, j:j + step_w] = tile

    # Blend with original to limit contrast amplification
    alpha = min(clip_limit / 3.0, 1.0)
    result = alpha * result + (1 - alpha) * arr

    return result


def tta_predict(
    model: Any,
    transform: Any,
    pil_image: Any,
    n_augments: int = 5,
) -> float:
    """
    Test-Time Augmentation (TTA) — average predictions over augmented views.

    Augmentations:
        - Original
        - Horizontal flip
        - Vertical flip
        - Small rotation (+5°, -5°)

    Returns averaged P(malignant).
    """
    if not _DEPS_OK:
        raise ImportError("torch required for TTA")

    img = pil_image.convert("RGB")
    augmentations = [
        img,                                          # original
        img.transpose(PILImage.FLIP_LEFT_RIGHT),      # h-flip
        img.transpose(PILImage.FLIP_TOP_BOTTOM),      # v-flip
        img.rotate(5, expand=False, fillcolor=0),     # +5° rotation
        img.rotate(-5, expand=False, fillcolor=0),    # -5° rotation
    ]

    probs = []
    model.eval()
    with torch.no_grad():
        for aug_img in augmentations[:n_augments]:
            tensor = transform(aug_img).unsqueeze(0)
            logits = model(tensor)
            prob = float(torch.softmax(logits, dim=1)[0, 1])
            probs.append(prob)

    avg_prob = float(np.mean(probs))
    logger.debug("TTA predictions: %s → avg=%.4f", probs, avg_prob)
    return avg_prob


def create_standard_transform(image_size: int = 224) -> Any:
    """Create the standard transform pipeline for EfficientNet inference."""
    if not _DEPS_OK:
        raise ImportError("torchvision required")

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
