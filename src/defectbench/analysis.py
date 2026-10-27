"""Detection operating points, PR curves and image bootstrap uncertainty."""
from __future__ import annotations

import numpy as np

from .metrics import image_metrics


def detection_analysis(labels, scores, threshold: float) -> dict:
    true = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=float)
    if true.ndim != 1 or true.shape != values.shape or not len(true) or not np.isfinite(values).all():
        raise ValueError("Need aligned, nonempty labels and finite scores")
    predicted = values >= threshold
    tp, fp = int((true & predicted).sum()), int((~true & predicted).sum())
    fn, tn = int((true & ~predicted).sum()), int((~true & ~predicted).sum())
    curve = []
    previous_recall = 0.0
    ap = 0.0
    positives = int(true.sum())
    for point in np.unique(values)[::-1]:
        selected = values >= point
        found = int((true & selected).sum())
        recall = found / positives if positives else 0.0
        precision = found / int(selected.sum())
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        curve.append({"threshold": float(point), "precision": precision, "recall": recall})
    return {
        **image_metrics(true, values, threshold),
        "threshold": threshold,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "average_precision": ap if positives else None,
        "precision_recall_curve": curve,
    }


def bootstrap_dice(rows: list[dict], *, draws: int = 1000, seed: int = 2026) -> dict:
    """Image-level bootstrap of micro Dice, not independent pixel resampling."""
    values = np.array([[r["tp"], r["fp"], r["fn"]] for r in rows], dtype=np.int64)
    if not len(values) or draws < 1:
        raise ValueError("Need image rows and positive draws")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), (draws, len(values)))
    totals = values[selected].sum(axis=1)
    denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
    dice = np.divide(2 * totals[:, 0], denominator, out=np.ones(draws), where=denominator > 0)
    low, high = np.quantile(dice, [0.025, 0.975])
    return {"unit": "image", "draws": draws, "seed": seed, "low": float(low), "high": float(high)}
