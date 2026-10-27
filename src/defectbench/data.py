"""Source image pairing and deterministic dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import cv2
import numpy as np


@dataclass(frozen=True)
class Sample:
    category: str
    image: Path
    mask: Path

    @property
    def key(self) -> str:
        return f"{self.category}/{self.image.stem}"


def read_grayscale(path: Path) -> np.ndarray:
    """Decode a grayscale image from a path that may contain Chinese characters."""
    if not path.is_file():
        raise ValueError(f"Image file missing: {path}")
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def discover_pairs(root: Path) -> list[Sample]:
    """Find the author's JPEG image and matching PNG mask for each sample."""
    samples: list[Sample] = []
    for image in sorted(root.rglob("*.jpg")):
        if image.parent.name != "Imgs" or not image.parent.parent.name.startswith("MT_"):
            continue
        mask = image.with_suffix(".png")
        if not mask.is_file():
            raise ValueError(f"Mask missing for {image}")
        samples.append(Sample(image.parent.parent.name, image, mask))
    if not samples:
        raise ValueError(f"No image/mask pairs found under {root}")
    if len({sample.key for sample in samples}) != len(samples):
        raise ValueError("Duplicate sample keys: keep only one copy of the source dataset")
    return samples


def split_samples(samples: list[Sample], seed: int = 42) -> dict[str, list[Sample]]:
    """Split each category 70/15/15 after sorting, then seeded shuffling."""
    if len({sample.key for sample in samples}) != len(samples):
        raise ValueError("Duplicate sample keys cannot be split without leakage")
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        groups.setdefault(sample.category, []).append(sample)
    result: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for category in sorted(groups):
        category_samples = sorted(groups[category], key=lambda sample: str(sample.image))
        Random(f"{seed}:{category}").shuffle(category_samples)
        count = len(category_samples)
        if count < 3:
            raise ValueError(f"Category {category} needs at least three samples")
        holdout = max(1, round(count * 0.15))
        result["val"].extend(category_samples[:holdout])
        result["test"].extend(category_samples[holdout : 2 * holdout])
        result["train"].extend(category_samples[2 * holdout :])
    for group in result.values():
        group.sort(key=lambda sample: sample.key)
    return result
