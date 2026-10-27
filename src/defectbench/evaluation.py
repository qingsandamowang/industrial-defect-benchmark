"""Validation-only threshold choice and full-resolution model inference."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .data import Sample, read_grayscale
from .metrics import image_metrics
from .preprocessing import letterbox, restore_map


def choose_mask_threshold(
    truths: list[np.ndarray], probabilities: list[np.ndarray], candidates: list[float]
) -> float:
    """Select the threshold with highest aggregate validation Dice."""
    if len(truths) != len(probabilities) or not truths or not candidates:
        raise ValueError("Need aligned validation masks and candidate thresholds")
    scores = []
    for threshold in candidates:
        tp = fp = fn = 0
        for truth, probability in zip(truths, probabilities):
            true = np.asarray(truth, dtype=bool)
            predicted = probability >= threshold
            tp += int(np.count_nonzero(true & predicted))
            fp += int(np.count_nonzero(~true & predicted))
            fn += int(np.count_nonzero(true & ~predicted))
        dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
        scores.append((dice, -abs(threshold - 0.5), threshold))
    return float(max(scores)[2])


def choose_image_threshold(
    labels: list[bool], scores: list[float], candidates: list[float]
) -> float:
    """Select an image detection threshold by validation F1."""
    if not labels or len(labels) != len(scores) or not candidates:
        raise ValueError("Need aligned image labels, scores and thresholds")
    ranked = [(image_metrics(labels, scores, threshold)["f1"], -abs(threshold - 0.5), threshold) for threshold in candidates]
    return float(max(ranked)[2])


def top_one_percent_score(probability: np.ndarray) -> float:
    """Mean of the most anomalous 1% of image pixels."""
    flat = np.asarray(probability).ravel()
    count = max(1, round(flat.size * 0.01))
    return float(np.partition(flat, -count)[-count:].mean())


def infer_probabilities(
    model: torch.nn.Module,
    samples: list[Sample],
    size: int = 128,
    device: str = "cpu",
) -> tuple[list[np.ndarray], list[float]]:
    """Run U-Net and map each probability image back to source resolution."""
    model.eval()
    maps: list[np.ndarray] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for sample in samples:
            source = read_grayscale(sample.image)
            prepared, geometry = letterbox(source, size)
            tensor = torch.from_numpy(prepared[None, None].astype(np.float32) / 255.0).to(device)
            start = perf_counter()
            probability = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
            latencies.append((perf_counter() - start) * 1000)
            maps.append(restore_map(probability, geometry))
    return maps, latencies
