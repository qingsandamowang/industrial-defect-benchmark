"""Run the real-data baseline and U-Net benchmark end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from .baseline import otsu_mask
from .data import Sample, discover_pairs, read_grayscale, split_samples
from .evaluation import choose_image_threshold, choose_mask_threshold, infer_probabilities, top_one_percent_score
from .reporting import summarize_predictions
from .training import TileDataset, fit_model


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _truths(samples: list[Sample]) -> list[np.ndarray]:
    return [read_grayscale(sample.mask) > 0 for sample in samples]


def _baseline_predict(samples: list[Sample], polarity: str, kernel: int, min_area: int) -> tuple[list[np.ndarray], list[float]]:
    predictions = []
    latencies = []
    for sample in samples:
        image = read_grayscale(sample.image)
        start = perf_counter()
        prediction = otsu_mask(image, polarity=polarity, kernel=kernel, min_area=min_area)
        latencies.append((perf_counter() - start) * 1000)
        predictions.append(prediction)
    return predictions, latencies


def _image_threshold(labels: list[bool], scores: list[float]) -> float:
    candidates = sorted(set(float(value) for value in np.linspace(0, 1, 101)) | set(scores))
    return choose_image_threshold(labels, scores, candidates)


def _baseline_run(groups: dict[str, list[Sample]], out: Path) -> dict:
    validation_truth = _truths(groups["val"])
    validation_labels = [bool(np.any(mask)) for mask in validation_truth]
    best: tuple[float, dict, list[np.ndarray]] | None = None
    for polarity in ("dark", "bright"):
        for kernel in (1, 3, 5):
            for min_area in (10, 30):
                predictions, _ = _baseline_predict(groups["val"], polarity, kernel, min_area)
                report = summarize_predictions(
                    validation_truth,
                    predictions,
                    [float(mask.mean()) for mask in predictions],
                    [sample.category for sample in groups["val"]],
                    0.01,
                )
                candidate = (report["pixel_micro_dice"], {"polarity": polarity, "kernel": kernel, "min_area": min_area}, predictions)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    assert best is not None
    _, parameters, validation_predictions = best
    validation_scores = [float(mask.mean()) for mask in validation_predictions]
    image_threshold = _image_threshold(validation_labels, validation_scores)
    test_truth = _truths(groups["test"])
    test_predictions, latencies = _baseline_predict(groups["test"], **parameters)
    test_scores = [float(mask.mean()) for mask in test_predictions]
    report = summarize_predictions(
        test_truth, test_predictions, test_scores, [sample.category for sample in groups["test"]], image_threshold
    )
    report.update(
        {
            "method": "OpenCV Otsu and morphology",
            "selection": "polarity/kernel/min_area by validation pixel Dice; image threshold by validation F1",
            "parameters": parameters,
            "image_score": "predicted foreground area fraction",
            "image_threshold": image_threshold,
            "median_inference_ms": statistics.median(latencies),
        }
    )
    _write_json(out / "baseline_report.json", report)
    return report


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    layer = bgr.copy()
    layer[mask] = color
    return cv2.addWeighted(bgr, 0.6, layer, 0.4, 0)


def _save_cases(samples: list[Sample], truths: list[np.ndarray], predictions: list[np.ndarray], out: Path) -> None:
    from .metrics import mask_metrics

    ranked = sorted(
        [(mask_metrics(truth, prediction)["dice"], index) for index, (truth, prediction) in enumerate(zip(truths, predictions)) if truth.any()]
    )[:12]
    normal_false_positives = [
        index for index, (truth, prediction) in enumerate(zip(truths, predictions)) if not truth.any() and prediction.any()
    ][:4]
    out.mkdir(parents=True, exist_ok=True)
    for _, index in ranked + [(0.0, index) for index in normal_false_positives]:
        sample = samples[index]
        image = read_grayscale(sample.image)
        panels = [
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
            _overlay(image, truths[index], (0, 180, 0)),
            _overlay(image, predictions[index], (0, 0, 220)),
        ]
        canvas = np.concatenate(panels, axis=1)
        target = out / f"{sample.category}_{sample.image.stem}.png"
        target.write_bytes(cv2.imencode(".png", canvas)[1].tobytes())


def _unet_run(groups: dict[str, list[Sample]], out: Path, epochs: int, size: int, batch_size: int) -> dict:
    train = TileDataset(groups["train"], size=size)
    validation = TileDataset(groups["val"], size=size)
    checkpoint = out / "unet_best.pt"
    model, history = fit_model(train, validation, checkpoint, epochs=epochs, batch_size=batch_size)
    _write_json(out / "unet_history.json", history)
    validation_truth = _truths(groups["val"])
    validation_probabilities, _ = infer_probabilities(model, groups["val"], size=size)
    mask_threshold = choose_mask_threshold(
        validation_truth, validation_probabilities, [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
    )
    validation_labels = [bool(np.any(mask)) for mask in validation_truth]
    validation_scores = [top_one_percent_score(probability) for probability in validation_probabilities]
    image_threshold = _image_threshold(validation_labels, validation_scores)
    test_truth = _truths(groups["test"])
    test_probabilities, latencies = infer_probabilities(model, groups["test"], size=size)
    test_predictions = [probability >= mask_threshold for probability in test_probabilities]
    test_scores = [top_one_percent_score(probability) for probability in test_probabilities]
    report = summarize_predictions(
        test_truth, test_predictions, test_scores, [sample.category for sample in groups["test"]], image_threshold
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report.update(
        {
            "method": "Lightweight U-Net with weighted BCE plus soft Dice",
            "selection": "best epoch and mask threshold by validation Dice; image threshold by validation F1",
            "image_size": size,
            "epochs_run": epochs,
            "best_epoch": saved["epoch"],
            "batch_size": batch_size,
            "mask_threshold": mask_threshold,
            "image_score": "mean probability of top 1% pixels",
            "image_threshold": image_threshold,
            "median_inference_ms": statistics.median(latencies),
        }
    )
    _write_json(out / "unet_report.json", report)
    _save_cases(groups["test"], test_truth, test_predictions, out / "case_images")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/mtd"))
    parser.add_argument("--source-zip", type=Path, default=Path("data/raw/mtd-source.zip"))
    parser.add_argument("--out", type=Path, default=Path("results/mtd_v1"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    samples = discover_pairs(args.data)
    groups = split_samples(samples, seed=42)
    manifest = {
        "source_url": "https://github.com/abin24/Magnetic-tile-defect-datasets.",
        "source_zip_sha256": hashlib.sha256(args.source_zip.read_bytes()).hexdigest(),
        "seed": 42,
        "splits": {name: [sample.key for sample in members] for name, members in groups.items()},
    }
    _write_json(args.out / "manifest.json", manifest)
    print("samples:", {name: len(members) for name, members in groups.items()}, flush=True)
    baseline = _baseline_run(groups, args.out)
    print(f"baseline: Dice={baseline['pixel_micro_dice']:.4f} image F1={baseline['image_f1']:.4f}", flush=True)
    unet = _unet_run(groups, args.out, args.epochs, args.size, args.batch_size)
    print(f"U-Net: Dice={unet['pixel_micro_dice']:.4f} image F1={unet['image_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
