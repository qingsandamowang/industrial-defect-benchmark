"""Aspect-preserving image preparation and prediction restoration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Geometry:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int


def letterbox(image: np.ndarray, size: int, mask: bool = False) -> tuple[np.ndarray, Geometry]:
    """Fit a grayscale image into a square while preserving its aspect ratio."""
    if image.ndim != 2 or min(image.shape) <= 0 or size <= 0:
        raise ValueError("Expected a nonempty grayscale image and positive size")
    height, width = image.shape
    scale = size / max(height, width)
    resized_height = max(1, round(height * scale))
    resized_width = max(1, round(width * scale))
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    fill = 0 if mask else int(np.median(image))
    canvas = np.full((size, size), fill, dtype=image.dtype)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, Geometry(height, width, resized_height, resized_width, top, left)


def restore_map(prepared: np.ndarray, geometry: Geometry, mask: bool = False) -> np.ndarray:
    """Remove letterbox padding and return a map at the source image resolution."""
    crop = prepared[
        geometry.top : geometry.top + geometry.resized_height,
        geometry.left : geometry.left + geometry.resized_width,
    ]
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    return cv2.resize(crop, (geometry.original_width, geometry.original_height), interpolation=interpolation)
