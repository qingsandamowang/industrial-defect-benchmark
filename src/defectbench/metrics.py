"""Pixel overlap and image-level defect detection metrics."""

from __future__ import annotations

import numpy as np


def mask_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Return confusion counts, IoU and Dice for binary masks."""
    if truth.shape != prediction.shape:
        raise ValueError(f"Mask shapes differ: {truth.shape} vs {prediction.shape}")
    true = np.asarray(truth, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = int(np.count_nonzero(true & pred))
    fp = int(np.count_nonzero(~true & pred))
    fn = int(np.count_nonzero(true & ~pred))
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    dice = (2 * tp) / (2 * tp + fp + fn) if union else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "dice": dice}


def image_metrics(
    truth: list[bool] | np.ndarray,
    scores: list[float] | np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    """Threshold image scores and compute precision, recall, F1, AUROC."""
    true = np.asarray(truth, dtype=bool)
    score = np.asarray(scores, dtype=float)
    if true.shape != score.shape or true.ndim != 1:
        raise ValueError("Image labels and scores must be matching one-dimensional arrays")
    predicted = score >= threshold
    tp = int(np.count_nonzero(true & predicted))
    fp = int(np.count_nonzero(~true & predicted))
    fn = int(np.count_nonzero(true & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    positive = score[true]
    negative = score[~true]
    auroc = None
    if len(positive) and len(negative):
        pairwise = positive[:, None] - negative[None, :]
        auroc = float((np.count_nonzero(pairwise > 0) + 0.5 * np.count_nonzero(pairwise == 0)) / pairwise.size)
    return {"precision": precision, "recall": recall, "f1": f1, "auroc": auroc}
