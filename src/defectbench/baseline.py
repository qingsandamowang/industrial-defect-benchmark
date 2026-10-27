"""Traditional thresholding baseline for comparison with U-Net."""

from __future__ import annotations

import cv2
import numpy as np


def otsu_mask(
    image: np.ndarray,
    polarity: str = "dark",
    kernel: int = 3,
    min_area: int = 10,
) -> np.ndarray:
    """Segment dark or bright regions using Otsu thresholding and cleanup."""
    if image.ndim != 2 or polarity not in {"dark", "bright"} or kernel < 1 or min_area < 1:
        raise ValueError("Invalid grayscale image or baseline parameters")
    if int(image.min()) == int(image.max()):
        return np.zeros(image.shape, dtype=bool)
    mode = cv2.THRESH_BINARY_INV if polarity == "dark" else cv2.THRESH_BINARY
    _, raw = cv2.threshold(image, 0, 255, mode | cv2.THRESH_OTSU)
    if kernel > 1:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, element)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    cleaned = np.zeros(raw.shape, dtype=bool)
    for index in range(1, component_count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == index] = True
    return cleaned
