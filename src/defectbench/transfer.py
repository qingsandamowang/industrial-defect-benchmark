"""Transfer segmentation model and explicit, portable checkpoint schema."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model import UNet


class DecoderBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        return self.block(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip), dim=1))


class ResNetUNet(nn.Module):
    """ImageNet ResNet18 features with a small U-shaped segmentation decoder.

    Input is one-channel grayscale in [0, 1]. Repetition to RGB and ImageNet
    normalization happen inside the model, including at deployment time.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1, self.layer2 = encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.dec4 = DecoderBlock(512 + 256, 128)
        self.dec3 = DecoderBlock(128 + 128, 64)
        self.dec2 = DecoderBlock(64 + 64, 32)
        self.dec1 = DecoderBlock(32 + 64, 16)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        spatial = image.shape[-2:]
        x = (image.expand(-1, 3, -1, -1) - self.mean) / self.std
        stem = self.stem(x)
        a = self.layer1(self.pool(stem))
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)
        x = self.dec4(d, c)
        x = self.dec3(x, b)
        x = self.dec2(x, a)
        x = self.dec1(x, stem)
        return F.interpolate(self.head(x), size=spatial, mode="bilinear", align_corners=False)


def create_model(architecture: str, *, pretrained: bool = False, base_channels: int = 8):
    if architecture == "resnet18_unet":
        return ResNetUNet(pretrained=pretrained)
    if architecture == "unet":
        return UNet(base_channels=base_channels)
    raise ValueError(f"Unsupported architecture: {architecture}")


def save_checkpoint(path: Path, model: nn.Module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "metadata": metadata, "model": model.state_dict()}, path)


def load_checkpoint(path: Path, device: str = "cpu"):
    """Use weights_only=True for local checkpoints; use trusted model files."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema_version") != 1 or not isinstance(saved.get("metadata"), dict):
        raise ValueError("Expected an experiment checkpoint with schema_version=1")
    metadata = saved["metadata"]
    size = metadata.get("size", 0)
    if not isinstance(size, int) or size < 32:
        raise ValueError("Checkpoint image size must be an integer >=32")
    for field in ("mask_threshold", "image_threshold"):
        if field in metadata and not 0 <= metadata[field] <= 1:
            raise ValueError(f"Invalid checkpoint {field}")
    model = create_model(metadata["architecture"], base_channels=metadata.get("base_channels", 8))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), metadata
