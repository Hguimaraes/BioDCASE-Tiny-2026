"""Tiny CNN models for 2D modulation-spectrum inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

if __name__ == "__main__":
  [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from pipeline_pytorch.model_base import ModelBase


class MSS2DTinyCNN(ModelBase):
  """Small CNN for [channel, acoustic_freq, modulation_freq] MSS maps."""

  def define_network_structure(self, n_filters=16, dropout=0.1):
    assert len(self.cfg["input_shape"]) == 3

    self.features = nn.Sequential(
      nn.Conv2d(self.cfg["input_shape"][0], n_filters, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(n_filters, n_filters * 2, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),

      nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3, stride=(2, 1), padding=1),
      nn.ReLU(),

      nn.Conv2d(n_filters * 4, n_filters * 4, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((1, 1)),
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(dropout),
      nn.Linear(n_filters * 4, 32),
      nn.ReLU(),
      nn.Linear(32, self.cfg["num_classes"]),
    )

  def forward(self, x):
    x = self.features(x)
    return self.classifier(x)
