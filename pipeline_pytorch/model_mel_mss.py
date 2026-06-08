"""Two-branch models for mel spectrogram plus MSS side-channel inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __name__ == "__main__":
  [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from pipeline_pytorch.model_base import ModelBase


class MelMSSSideChannelCNN(ModelBase):
  """Late-fusion CNN with a baseline-like mel branch and compact MSS branch."""

  def define_network_structure(self):
    mel_shape = tuple(self.cfg["mel_shape"])
    mss_shape = tuple(self.cfg["mss_shape"])
    assert len(mel_shape) == 3
    assert len(mss_shape) == 3
    self.mel_shape = mel_shape
    self.mss_shape = mss_shape
    self.mel_size = int(torch.zeros(mel_shape).numel())
    self.mss_size = int(torch.zeros(mss_shape).numel())
    self.expected_input_size = self.mel_size + self.mss_size

    mel_filters = self.cfg.get("mel_filters", 32)
    mss_filters = self.cfg.get("mss_filters", 8)
    dropout = self.cfg.get("dropout", 0.2)

    self.mel_branch = nn.Sequential(
      nn.Conv2d(mel_shape[0], mel_filters, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(mel_filters, mel_filters * 2, kernel_size=3),
      nn.ReLU(),
      nn.MaxPool2d(4),

      nn.Conv2d(mel_filters * 2, mel_filters * 4, kernel_size=3),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
    )

    self.mss_branch = nn.Sequential(
      nn.Conv2d(mss_shape[0], mss_filters, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(mss_filters, mss_filters * 2, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(mss_filters * 2, mss_filters * 4, kernel_size=3, stride=(2, 1), padding=1),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
    )

    fusion_dim = (mel_filters * 4) + (mss_filters * 4)
    hidden_dim = self.cfg.get("fusion_hidden_dim", 64)
    self.classifier = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(fusion_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, self.cfg["num_classes"]),
    )

  def split_features(self, x):
    if x.ndim > 2:
      x = x.flatten(start_dim=1)
    if x.shape[1] != self.expected_input_size:
      raise ValueError(
        f"Expected flat input size {self.expected_input_size}, got {x.shape[1]}"
      )
    mel = x[:, : self.mel_size].reshape((-1,) + self.mel_shape)
    mss = x[:, self.mel_size :].reshape((-1,) + self.mss_shape)
    return mel, mss

  def forward(self, x):
    mel, mss = self.split_features(x)
    mel_embedding = self.mel_branch(mel)
    mss_embedding = self.mss_branch(mss)
    fused = torch.cat([mel_embedding, mss_embedding], dim=1)
    return self.classifier(fused)
