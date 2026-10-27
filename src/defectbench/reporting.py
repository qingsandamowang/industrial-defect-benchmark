"""Aggregate pixel and image metrics without including true negatives in overlap."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import image_metrics, mask_metrics


def summarize_predictions(
    truths: list[np.ndarray],
    predictions: list[np.ndarray],
    scores: list[float],
    categories: list[str],
    image_threshold: float,
) -> dict:
    if not (len(truths) == len(predictions) == len(scores) == len(categories)) or not truths:
        raise ValueError("Prediction inputs must be nonempty and aligned")
    rows = [mask_metrics(truth, prediction) for truth, prediction in zip(truths, predictions)]
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    union = tp + fp + fn
    pixel_iou = tp / union if union else 1.0
    pixel_dice = 2 * tp / (2 * tp + fp + fn) if union else 1.0
    labels = [bool(np.any(truth)) for truth in truths]
    image = image_metrics(labels, scores, image_threshold)
    category_rows: dict[str, list[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        category_rows[category].append(index)
    per_category = {}
    for category, indices in sorted(category_rows.items()):
        class_tp = sum(rows[index]["tp"] for index in indices)
        class_fp = sum(rows[index]["fp"] for index in indices)
        class_fn = sum(rows[index]["fn"] for index in indices)
        class_union = class_tp + class_fp + class_fn
        per_category[category] = {
            "count": len(indices),
            "positive_masks": sum(labels[index] for index in indices),
            "pixel_micro_iou": class_tp / class_union if class_union else 1.0,
            "pixel_micro_dice": 2 * class_tp / (2 * class_tp + class_fp + class_fn) if class_union else 1.0,
            "image_f1": image_metrics([labels[index] for index in indices], [scores[index] for index in indices], image_threshold)["f1"],
        }
    return {
        "count": len(truths),
        "positive_masks": sum(labels),
        "pixel_micro_iou": pixel_iou,
        "pixel_micro_dice": pixel_dice,
        "pixel_tp": tp,
        "pixel_fp": fp,
        "pixel_fn": fn,
        "image_precision": image["precision"],
        "image_recall": image["recall"],
        "image_f1": image["f1"],
        "image_auroc": image["auroc"],
        "per_category": per_category,
    }
