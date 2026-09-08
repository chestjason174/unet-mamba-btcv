from __future__ import annotations

import torch
import torch.nn as nn


NUM_CLASSES = 14


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, base_channels=8, debug_shapes=False):
        super().__init__()
        c = base_channels
        self.debug_shapes = debug_shapes

        self.enc1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(c * 2, c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)

        self.out = nn.Conv3d(c, num_classes, kernel_size=1)

    def _print_shape(self, name, x):
        if self.debug_shapes:
            print(f"[model] {name}: {tuple(x.shape)}")

    def forward(self, x):
        self._print_shape("input", x)
        e1 = self.enc1(x)
        self._print_shape("enc1", e1)
        e2 = self.enc2(self.pool1(e1))
        self._print_shape("enc2", e2)
        b = self.bottleneck(self.pool2(e2))
        self._print_shape("bottleneck", b)

        d2 = self.up2(b)
        self._print_shape("up2", d2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        self._print_shape("dec2", d2)

        d1 = self.up1(d2)
        self._print_shape("up1", d1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        self._print_shape("dec1", d1)

        logits = self.out(d1)
        self._print_shape("logits", logits)
        return logits
