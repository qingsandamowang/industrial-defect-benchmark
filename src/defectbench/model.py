"""Lightweight encoder-decoder network for binary segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 8):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(1, c)
        self.enc2 = DoubleConv(c, 2 * c)
        self.enc3 = DoubleConv(2 * c, 4 * c)
        self.bottleneck = DoubleConv(4 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = DoubleConv(8 * c + 4 * c, 4 * c)
        self.dec2 = DoubleConv(4 * c + 2 * c, 2 * c)
        self.dec1 = DoubleConv(2 * c + c, c)
        self.head = nn.Conv2d(c, 1, 1)

    @staticmethod
    def _up_join(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((x, skip), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        middle = self.bottleneck(self.pool(c))
        up = self.dec3(self._up_join(middle, c))
        up = self.dec2(self._up_join(up, b))
        up = self.dec1(self._up_join(up, a))
        return self.head(up)
