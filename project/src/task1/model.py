"""Small CNN baseline for 1x64x64 Micro-C contact windows."""

from __future__ import annotations

import torch
from torch import nn


class SmallMicroCCNN(nn.Module):
    def __init__(self, dropout: float = 0.3, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    @property
    def target_layer(self) -> nn.Conv2d:
        return self.features[8]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier[3:](self.forward_features(inputs))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the stable post-ReLU hidden vector before dropout and logits."""
        return self.classifier[:3](self.pool(self.features(inputs)))
