"""Predict arbitrary images with an explicit experiment checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from .data import read_grayscale
from .evaluation import top_one_percent_score
from .html_report import write_prediction_index
from .preprocessing import letterbox, restore_map
from .transfer import load_checkpoint

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_output_location(source: Path, output: Path) -> None:
    source, output = source.resolve(), output.resolve()
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("output must be outside the input directory")
    if source.is_file() and source.parent == output:
        raise ValueError("output must differ from the input image directory")


def predict_probability(model, source: np.ndarray, size: int, device: str = "cpu") -> np.ndarray:
    prepared, geometry = letterbox(source, size)
    tensor = torch.from_numpy(prepared[None, None].astype(np.float32) / 255.0).to(device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
    return restore_map(probability, geometry)


def regions_from_mask(mask: np.ndarray, probability: np.ndarray) -> list[dict]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        regions.append({"bbox_xywh": [x, y, width, height], "area_px": area, "mean_probability": float(probability[labels == label].mean())})
    return sorted(regions, key=lambda region: region["area_px"], reverse=True)


def predict_inputs(checkpoint: Path, source: Path, output: Path, device: str = "cpu") -> dict:
    source, output = Path(source), Path(output)
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    validate_output_location(source, output)
    paths = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS and not path.is_symlink()) if source.is_dir() else [source]
    if not paths or any(path.suffix.lower() not in EXTENSIONS for path in paths):
        raise ValueError("No supported input images")
    prefixes = [path.relative_to(source).with_suffix("") if source.is_dir() else Path(path.stem) for path in paths]
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("Input images share an output stem; rename duplicate stems")
    model, metadata = load_checkpoint(checkpoint, device=device)
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for path, prefix in zip(paths, prefixes):
        image = read_grayscale(path)
        start = perf_counter()
        probability = predict_probability(model, image, metadata["size"], device)
        elapsed = (perf_counter() - start) * 1000
        mask = probability >= metadata.get("mask_threshold", 0.5)
        score = top_one_percent_score(probability)
        target = output / prefix
        target.parent.mkdir(parents=True, exist_ok=True)
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        color = overlay.copy()
        color[mask] = (0, 0, 255)
        overlay = cv2.addWeighted(overlay, 0.6, color, 0.4, 0)
        (target.parent / (target.name + "_mask.png")).write_bytes(cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tobytes())
        (target.parent / (target.name + "_overlay.png")).write_bytes(cv2.imencode(".png", overlay)[1].tobytes())
        np.save(target.parent / (target.name + "_probability.npy"), probability, allow_pickle=False)
        items.append({"file": path.relative_to(source).as_posix() if source.is_dir() else path.name, "width": image.shape[1], "height": image.shape[0], "predicted_defect": score >= metadata.get("image_threshold", 0.5), "image_score": score, "defect_area_px": int(mask.sum()), "regions": regions_from_mask(mask, probability), "latency_ms": elapsed})
    result = {"count": len(items), "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(), "metadata": metadata, "items": items, "note": "No ground truth supplied: predictions do not establish accuracy; probabilities are uncalibrated model scores."}
    (output / "predictions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_prediction_index(output / "index.html", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    try:
        result = predict_inputs(args.checkpoint, args.input, args.out, args.device)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Predicted {result['count']} image(s); outputs: {args.out}")


if __name__ == "__main__":
    main()
