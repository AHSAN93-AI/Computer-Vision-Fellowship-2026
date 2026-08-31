"""
Decode an uploaded image and turn it into the tensor shape both models
expect: grayscale, IMG_SIZE x IMG_SIZE, normalized to [0, 1].
"""

import cv2
import numpy as np
import torch

from app.config import IMG_SIZE


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Bytes -> full-resolution grayscale uint8 numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image -- unsupported or corrupt file")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def to_model_tensor(gray_image: np.ndarray) -> torch.Tensor:
    """Full-res grayscale array -> (1, 1, IMG_SIZE, IMG_SIZE) float tensor in [0, 1]."""
    resized = cv2.resize(gray_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


def resized_bgr_for_display(gray_image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Small BGR copy at model resolution, used as the base for heatmap overlays."""
    resized = cv2.resize(gray_image, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
