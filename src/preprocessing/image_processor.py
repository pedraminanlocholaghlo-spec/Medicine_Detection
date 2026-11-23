"""Image loading, validation, resizing and configurable lightweight enhancement."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path) -> np.ndarray:
    """Read a JPG/PNG from disk (np.fromfile keeps non-ASCII paths working)."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Image not found: {path.name}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path.name}")
    return image


def decode_image_bytes(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid JPG or PNG image")
    return image


def validate_extension(filename: str | None, allowed: tuple) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise ValueError(f"Unsupported file type '{suffix or 'unknown'}'. Allowed: {', '.join(allowed)}")
    return suffix


def validate_size(data: bytes, max_mb: float) -> None:
    if not data:
        raise ValueError("Uploaded file is empty")
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"File is larger than {max_mb:g} MB")


def validate_image(image: np.ndarray, min_side: int = 120) -> np.ndarray:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be a color JPG or PNG")
    height, width = image.shape[:2]
    if min(height, width) < min_side:
        raise ValueError(f"Image is too small ({width}x{height}); minimum {min_side}px per side")
    return image


def resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return cv2.resize(
        image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


ENHANCEMENTS = {"clahe": enhance_contrast, "sharpen": sharpen}


def preprocess(image: np.ndarray, config) -> tuple[np.ndarray, list[str]]:
    """Resize (always) and optionally enhance. Returns (image, applied_ops)."""
    processed = resize_long_side(image, config.max_image_side)
    applied = ["resize"]
    if getattr(config, "enable_enhancement", False):
        for name in config.enhancement_ops:
            operation = ENHANCEMENTS.get(name)
            if operation is None:
                raise ValueError(f"Unknown enhancement operation: {name}")
            processed = operation(processed)
            applied.append(name)
    return processed, applied
