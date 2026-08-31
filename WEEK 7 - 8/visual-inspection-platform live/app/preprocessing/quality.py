"""
Requirement 18: Image Quality Validation.

Runs before any model inference. If the image fails these checks, the
pipeline short-circuits with INSPECTION INVALID and never wastes a model
pass on unusable data.
"""

import cv2
import numpy as np

from app.config import QUALITY_THRESHOLDS


def check_image_quality(gray_image: np.ndarray) -> dict:
    """
    gray_image: 2D uint8 numpy array (grayscale, original resolution --
    run this BEFORE resizing to the model's 64x64 input so blur/brightness
    reflect the actual capture, not a downsampled version of it).

    Returns:
        {"valid": bool, "reason": str | None, "brightness": float, "blur_score": float}
    """
    brightness = float(gray_image.mean())
    blur_score = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    if brightness < QUALITY_THRESHOLDS["min_brightness"]:
        return _invalid("Image too dark", brightness, blur_score)
    if brightness > QUALITY_THRESHOLDS["max_brightness"]:
        return _invalid("Image too bright", brightness, blur_score)
    if blur_score < QUALITY_THRESHOLDS["min_laplacian_var"]:
        return _invalid("Image too blurred", brightness, blur_score)

    return {"valid": True, "reason": None, "brightness": brightness, "blur_score": blur_score}


def _invalid(reason, brightness, blur_score):
    return {"valid": False, "reason": reason, "brightness": brightness, "blur_score": blur_score}
