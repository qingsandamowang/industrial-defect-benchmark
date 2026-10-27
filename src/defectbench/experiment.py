"""Train on train/validation only; evaluate heldout images in a separate command."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .analysis import bootstrap_dice, detection_analysis
from .data import discover_pairs, read_grayscale, split_samples
from .evaluation import choose_image_threshold, choose_mask_threshold, infer_probabilities, top_one_percent_score
from .html_report import write_report
from .metrics import mask_metrics
from .reporting import summarize_predictions
from .training import TileDataset
from .transfer import create_model, load_checkpoint, save_checkpoint


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def content_manifest(groups):
    """Register file bytes for provenance, without decoding heldout pixels."""
    samples = {sample.key: sample for group in groups.values() for sample in group}
    rows = [{"key": key, "image_sha256": hashlib.sha256(sample.image.read_bytes()).hexdigest(), "mask_sha256": hashlib.sha256(sample.mask.read_bytes()).hexdigest()} for key, sample in sorted(samples.items())]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"dataset_sha256": hashlib.sha256(encoded).hexdigest(), "samples": rows}


def paired_augment(image, mask, *, photometric=True):
    image, mask = image.clone(), mask.clone()
    if torch.rand(()) < 0.5:
        image, mask = image.flip(-1), mask.flip(-1)
    if torch.rand(()) < 0.5:
        image, mask = image.flip(-2), mask.flip(-2)
    if photometric:
        image = (image * (0.8 + 0.4 * torch.rand(())) + (torch.rand(()) - 0.5) * 0.1).clamp(0, 1)
        if torch.rand(()) < 0.25:
            image = (image + torch.randn_like(image) * 0.015).clamp(0, 1)
    return image.contiguous(), mask.contiguous()


class AugmentedDataset(Dataset):
    def __init__(self, base: TileDataset):
        self.base = base
        self.has_defect = base.has_defect

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return paired_augment(*self.base[index])


def load_training_data(data: Path, size: int, seed: int = 42):
    groups = split_samples(discover_pairs(data), seed=seed)
    # Heldout files are listed but their pixels are not opened by training.
    train = AugmentedDataset(TileDataset(groups["train"], size=size))
    validation = TileDataset(groups["val"], size=size)
    return train, validation, groups


def segmentation_loss(logits, target):
    # At 384x384, FP16 spatial sums can exceed 65504 even for p=0.5.
    # Keep both BCE and overlap reductions in FP32 while preserving gradients.
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    dice = (2 * intersection + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + (1 - dice).mean()


def validation_selection(model, samples, size, device):
    probabilities, _ = infer_probabilities(model, samples, size=size, device=device)
    truths = [read_grayscale(sample.mask) > 0 for sample in samples]
    mask_threshold = choose_mask_threshold(truths, probabilities, [round(float(x), 2) for x in np.arange(0.1, 0.91, 0.05)])
    scores = [top_one_percent_score(p) for p in probabilities]
    labels = [bool(truth.any()) for truth in truths]
    image_threshold = choose_image_threshold(labels, scores, sorted(set(scores) | set(np.linspace(0, 1, 101))))
    report = summarize_predictions(truths, [p >= mask_threshold for p in probabilities], scores, [s.category for s in samples], image_threshold)
    return mask_threshold, image_threshold, report


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32 or args.patience < 1:
        raise ValueError("Positive epochs, batch size, patience and image size >=32 required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable: install the matching GPU PyTorch build or choose --device cpu")
    if (args.out / "config.json").exists():
        raise ValueError("Experiment already exists; choose a new --out to preserve its records")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset, _, groups = load_training_data(args.data, args.size, seed=args.split_seed)
    config = {key: value for key, value in vars(args).items() if key not in {"data", "out", "command"}}
    config.update({"python": platform.python_version(), "torch": str(torch.__version__), "opencv": cv2.__version__, "hardware": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(), "augmentation": "paired H/V flips, brightness/contrast, Gaussian noise", "loss": "unweighted BCE + per-image soft Dice; reductions in FP32", "sampling": "balanced image-level positive/negative", "test_access": "compressed file bytes hashed only for provenance; heldout pixels never decoded, optimized or scored by train"})
    write_json(args.out / "config.json", config)
    splits = {name: [sample.key for sample in members] for name, members in groups.items()}
    write_json(args.out / "manifest.json", {"split_seed": args.split_seed, "splits": splits})
    fingerprint = content_manifest(groups)
    write_json(args.out / "data_fingerprint.json", fingerprint)
    model = create_model(args.architecture, pretrained=not args.no_pretrained, base_channels=args.base_channels).to(args.device)
    positives = sum(dataset.has_defect)
    negatives = len(dataset) - positives
    weights = [1 / positives if positive else 1 / negatives for positive in dataset.has_defect] if positives and negatives else [1.0] * len(dataset)
    sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=args.device == "cuda")
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith(("stem.", "layer")) else decoder).append(parameter)
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.learning_rate * 0.2}, {"params": decoder, "lr": args.learning_rate}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_dice, best_epoch = -1.0, 0
    checkpoint = args.out / "model_best.pt"
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        total_loss, seen = 0.0, 0
        for image, truth in loader:
            image, truth = image.to(args.device), truth.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=use_amp):
                loss = segmentation_loss(model(image), truth)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; experiment stopped before publishing metrics")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(image)
            seen += len(image)
        scheduler.step()
        mask_threshold, image_threshold, report = validation_selection(model, groups["val"], args.size, args.device)
        dice = report["pixel_micro_dice"]
        history.append({"epoch": epoch, "train_loss": total_loss / seen, "validation_dice": dice, "validation_image_f1": report["image_f1"], "mask_threshold": mask_threshold, "image_threshold": image_threshold, "seconds": perf_counter() - start})
        write_json(args.out / "history.json", history)
        print(f"epoch {epoch}/{args.epochs} loss={total_loss / seen:.4f} val_dice={dice:.4f} val_F1={report['image_f1']:.4f} time={history[-1]['seconds']:.1f}s", flush=True)
        if dice > best_dice:
            best_dice, best_epoch = dice, epoch
            metadata = {"architecture": args.architecture, "base_channels": args.base_channels, "size": args.size, "mask_threshold": mask_threshold, "image_threshold": image_threshold, "seed": args.seed, "split_seed": args.split_seed, "epoch": epoch, "validation_dice": dice, "pretrained_source": "torchvision ResNet18 IMAGENET1K_V1" if args.architecture == "resnet18_unet" and not args.no_pretrained else None, "splits": splits, "dataset_sha256": fingerprint["dataset_sha256"]}
            save_checkpoint(checkpoint, model, metadata)
            write_json(args.out / "validation_report.json", report)
        if epoch - best_epoch >= args.patience:
            print("Early stopping: validation Dice did not improve", flush=True)
            break
    write_json(args.out / "candidate.json", {"best_epoch": best_epoch, "validation_dice": best_dice, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "epochs_run": len(history), "selection": "maximum validation pixel micro Dice, thresholds selected only on validation", "test_evaluated": False})


def evaluate(args):
    if (args.out / "test_report.json").exists():
        raise ValueError("Test report already exists; choose a separate output path")
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    groups = split_samples(discover_pairs(args.data), seed=metadata["split_seed"])
    actual_splits = {name: [sample.key for sample in members] for name, members in groups.items()}
    if actual_splits != metadata["splits"]:
        raise ValueError("Dataset keys differ from the saved split; heldout evaluation refused")
    fingerprint = content_manifest(groups)
    if fingerprint["dataset_sha256"] != metadata.get("dataset_sha256"):
        raise ValueError("Dataset content differs from training registration; heldout evaluation refused")
    fingerprints = {row["key"]: row for row in fingerprint["samples"]}
    samples = groups["test"]
    truths = [read_grayscale(sample.mask) > 0 for sample in samples]
    probabilities, latencies = infer_probabilities(model, samples, size=metadata["size"], device=args.device)
    predictions = [p >= metadata["mask_threshold"] for p in probabilities]
    scores = [top_one_percent_score(p) for p in probabilities]
    report = summarize_predictions(truths, predictions, scores, [s.category for s in samples], metadata["image_threshold"])
    rows = [{"key": sample.key, "category": sample.category, "has_defect": bool(truth.any()), "image_score": score, **fingerprints[sample.key], **mask_metrics(truth, predicted)} for sample, truth, predicted, score in zip(samples, truths, predictions, scores)]
    report["detection"] = detection_analysis([bool(t.any()) for t in truths], scores, metadata["image_threshold"])
    report["dice_bootstrap_95_ci"] = bootstrap_dice(rows)
    report["experiment"] = {key: value for key, value in metadata.items() if key != "splits"}
    report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report["median_inference_ms"] = float(np.median(latencies))
    write_json(args.out / "test_report.json", report)
    write_json(args.out / "per_image.json", rows)
    write_json(args.out / "manifest.json", {"splits": actual_splits, "split_seed": metadata["split_seed"], "dataset_sha256": fingerprint["dataset_sha256"], "test_selection": "fixed heldout split; checkpoint and thresholds frozen before this command"})
    write_report(args.out / "report.html", report)
    from .run import _save_cases
    _save_cases(samples, truths, predictions, args.out / "case_images")
    print(f"HELDOUT Dice={report['pixel_micro_dice']:.4f} F1={report['image_f1']:.4f} AUROC={report['image_auroc']:.4f} FPR={report['detection']['false_positive_rate']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("data/raw/mtd"))
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=40)
    training.add_argument("--size", type=int, default=384)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--architecture", choices=("resnet18_unet", "unet"), default="resnet18_unet")
    training.add_argument("--base-channels", type=int, default=16)
    training.add_argument("--learning-rate", type=float, default=1e-3)
    training.add_argument("--patience", type=int, default=12)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--split-seed", type=int, default=42)
    training.add_argument("--no-pretrained", action="store_true")
    training.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, default=Path("data/raw/mtd"))
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        train(args) if args.command == "train" else evaluate(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
