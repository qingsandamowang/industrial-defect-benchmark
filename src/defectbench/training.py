"""Data loading and deterministic lightweight U-Net training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import Sample, read_grayscale
from .model import UNet
from .preprocessing import letterbox


class TileDataset(Dataset):
    def __init__(self, samples: list[Sample], size: int = 128):
        self.images: list[torch.Tensor] = []
        self.masks: list[torch.Tensor] = []
        self.has_defect: list[bool] = []
        for sample in samples:
            image = read_grayscale(sample.image)
            mask = read_grayscale(sample.mask)
            if image.shape != mask.shape:
                raise ValueError(f"Unreadable or mismatched pair: {sample.key}")
            self.has_defect.append(bool(np.any(mask)))
            prepared_image, _ = letterbox(image, size)
            prepared_mask, _ = letterbox(mask, size, mask=True)
            self.images.append(torch.from_numpy(prepared_image[None].astype(np.float32) / 255.0))
            self.masks.append(torch.from_numpy((prepared_mask[None] > 0).astype(np.float32)))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.masks[index]


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Class-weighted BCE plus soft Dice loss for sparse defects."""
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=logits.new_tensor(5.0))
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss


def _validation_dice(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    tp = fp = fn = 0
    model.eval()
    with torch.inference_mode():
        for image, truth in loader:
            predicted = torch.sigmoid(model(image.to(device))) >= 0.5
            true = truth.to(device) > 0.5
            tp += int((predicted & true).sum())
            fp += int((predicted & ~true).sum())
            fn += int((~predicted & true).sum())
    return (2 * tp) / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def fit_model(
    train: Dataset,
    validation: Dataset,
    checkpoint: Path,
    epochs: int = 12,
    batch_size: int = 16,
    base_channels: int = 8,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> tuple[UNet, list[dict[str, float | int]]]:
    """Fit on training images and retain the epoch with best validation Dice."""
    if epochs < 1 or batch_size < 1 or len(train) < 1 or len(validation) < 1:
        raise ValueError("Need nonempty datasets, positive epochs and batch size")
    target_device = torch.device(device)
    model = UNet(base_channels=base_channels).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    positives = sum(train.has_defect)
    negatives = len(train) - positives
    if positives and negatives:
        weights = [1 / positives if defect else 1 / negatives for defect in train.has_defect]
        sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
        train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for image, truth in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(image.to(target_device)), truth.to(target_device))
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(image)
        validation_dice = _validation_dice(model, validation_loader, target_device)
        record = {"epoch": epoch, "train_loss": loss_total / len(train), "validation_dice_at_0_5": validation_dice}
        history.append(record)
        print(f"epoch {epoch}/{epochs}: loss={record['train_loss']:.4f} val_dice={validation_dice:.4f}", flush=True)
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save({"epoch": epoch, "model": model.state_dict(), "base_channels": base_channels}, checkpoint)
    saved = torch.load(checkpoint, map_location=target_device, weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, history
