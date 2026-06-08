"""Two-branch models for mel spectrogram plus MSS side-channel inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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


class MelMSSLogitDistillationCNN(MelMSSSideChannelCNN):
  """Mel+MSS student trained with hard labels plus teacher logit distillation."""

  def distillation_cfg(self):
    return {
      "hard_loss_weight": 1.0,
      "soft_loss_weight": 0.5,
      "temperature": 2.0,
      "gradient_clip_norm": None,
      **self.cfg.get("distillation", {}),
    }

  def compute_losses(self, y_hat, y, teacher_logits=None):
    hard_loss = self.criterion(y_hat, y)
    cfg = self.distillation_cfg()
    soft_weight = float(cfg["soft_loss_weight"])
    if teacher_logits is None or soft_weight <= 0:
      return hard_loss, {
        "total_loss": float(hard_loss.detach().cpu()),
        "hard_loss": float(hard_loss.detach().cpu()),
        "soft_loss": 0.0,
      }

    temperature = float(cfg["temperature"])
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(y_hat / temperature, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    total_loss = float(cfg["hard_loss_weight"]) * hard_loss + soft_weight * soft_loss
    return total_loss, {
      "total_loss": float(total_loss.detach().cpu()),
      "hard_loss": float(hard_loss.detach().cpu()),
      "soft_loss": float(soft_loss.detach().cpu()),
    }

  def train_step(self, data):
    self.optimizer.zero_grad()

    x = data[0].to(device=self.device, dtype=torch.float32)
    y = data[1].to(device=self.device)
    teacher_logits = None
    if len(data) > 3:
      teacher_logits = data[3].to(device=self.device, dtype=torch.float32)

    y_hat = self.forward(x)
    loss, metrics = self.compute_losses(y_hat, y, teacher_logits)
    loss.backward()

    clip_norm = self.distillation_cfg().get("gradient_clip_norm")
    if clip_norm is not None:
      torch.nn.utils.clip_grad_norm_(self.parameters(), float(clip_norm))

    self.optimizer.step()
    self.last_train_step_metrics = metrics
    return loss.item()

  def validation_step(self, data):
    with torch.no_grad():
      x = data[0].to(device=self.device, dtype=torch.float32)
      y = data[1].to(device=self.device)
      teacher_logits = None
      if len(data) > 3:
        teacher_logits = data[3].to(device=self.device, dtype=torch.float32)

      y_hat = self.forward(x)
      loss, metrics = self.compute_losses(y_hat, y, teacher_logits)
      self.last_validation_step_metrics = metrics
      y_hat = self.prediction_post_processing(y_hat)

    return y_hat, loss.item()


class MelMSSSEDLogitDistillationCNN(MelMSSLogitDistillationCNN):
  """Tiny SED-style Mel+MSS student with Perch logit distillation.

  The mel branch keeps the time axis and predicts framewise logits. A compact
  MSS branch provides global context that is broadcast across mel frames before
  attention pooling.
  """

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

    mel_filters = self.cfg.get("mel_filters", 16)
    mel_temporal_dim = self.cfg.get("mel_temporal_dim", 96)
    mss_filters = self.cfg.get("mss_filters", 8)
    mss_context_dim = self.cfg.get("mss_context_dim", 32)
    sed_hidden_dim = self.cfg.get("sed_hidden_dim", 96)
    dropout = self.cfg.get("dropout", 0.25)

    self.mel_encoder = nn.Sequential(
      nn.Conv2d(mel_shape[0], mel_filters, kernel_size=3, padding=1),
      nn.BatchNorm2d(mel_filters),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 1)),

      nn.Conv2d(mel_filters, mel_filters * 2, kernel_size=3, padding=1),
      nn.BatchNorm2d(mel_filters * 2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 1)),

      nn.Conv2d(mel_filters * 2, mel_temporal_dim, kernel_size=3, padding=1),
      nn.BatchNorm2d(mel_temporal_dim),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 1)),

      nn.Conv2d(mel_temporal_dim, mel_temporal_dim, kernel_size=3, padding=1),
      nn.BatchNorm2d(mel_temporal_dim),
      nn.ReLU(),
    )

    self.mss_branch = nn.Sequential(
      nn.Conv2d(mss_shape[0], mss_filters, kernel_size=5, stride=2, padding=2),
      nn.BatchNorm2d(mss_filters),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(mss_filters, mss_filters * 2, kernel_size=3, padding=1),
      nn.BatchNorm2d(mss_filters * 2),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(mss_filters * 2, mss_filters * 4, kernel_size=3, stride=(2, 1), padding=1),
      nn.BatchNorm2d(mss_filters * 4),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(mss_filters * 4, mss_context_dim),
      nn.ReLU(),
    )

    self.frame_fusion = nn.Sequential(
      nn.Dropout(dropout),
      nn.Conv1d(mel_temporal_dim + mss_context_dim, sed_hidden_dim, kernel_size=1),
      nn.ReLU(),
      nn.Dropout(dropout),
    )
    self.attention = nn.Conv1d(sed_hidden_dim, self.cfg["num_classes"], kernel_size=1)
    self.classifier = nn.Conv1d(sed_hidden_dim, self.cfg["num_classes"], kernel_size=1)

  def encode_frame_features(self, x):
    mel, mss = self.split_features(x)
    mel_features = self.mel_encoder(mel).mean(dim=2)
    mss_context = self.mss_branch(mss).unsqueeze(-1).expand(-1, -1, mel_features.shape[-1])
    return self.frame_fusion(torch.cat([mel_features, mss_context], dim=1))

  def forward(self, x, return_framewise=False):
    frame_features = self.encode_frame_features(x)
    attention = torch.softmax(torch.tanh(self.attention(frame_features)), dim=-1)
    framewise_logits = self.classifier(frame_features)
    clip_logits = torch.sum(attention * framewise_logits, dim=-1)
    if return_framewise:
      return clip_logits, framewise_logits
    return clip_logits

  def compute_sed_losses(self, clip_logits, framewise_logits, y, teacher_logits=None):
    cfg = self.distillation_cfg()
    frame_max_logits = framewise_logits.max(dim=-1).values
    clip_hard_loss = self.criterion(clip_logits, y)
    frame_hard_loss = self.criterion(frame_max_logits, y)
    frame_loss_weight = float(cfg.get("frame_loss_weight", 0.5))
    hard_loss = (1.0 - frame_loss_weight) * clip_hard_loss + frame_loss_weight * frame_hard_loss

    soft_weight = float(cfg["soft_loss_weight"])
    soft_loss = clip_logits.new_tensor(0.0)
    if teacher_logits is not None and soft_weight > 0:
      temperature = float(cfg["temperature"])
      teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
      clip_log_probs = F.log_softmax(clip_logits / temperature, dim=1)
      clip_soft_loss = F.kl_div(clip_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
      frame_soft_loss = clip_soft_loss
      if cfg.get("distill_frame_max", True):
        frame_log_probs = F.log_softmax(frame_max_logits / temperature, dim=1)
        frame_soft_loss = F.kl_div(frame_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
      soft_loss = (1.0 - frame_loss_weight) * clip_soft_loss + frame_loss_weight * frame_soft_loss

    total_loss = float(cfg["hard_loss_weight"]) * hard_loss + soft_weight * soft_loss
    return total_loss, {
      "total_loss": float(total_loss.detach().cpu()),
      "hard_loss": float(hard_loss.detach().cpu()),
      "soft_loss": float(soft_loss.detach().cpu()),
      "clip_hard_loss": float(clip_hard_loss.detach().cpu()),
      "frame_hard_loss": float(frame_hard_loss.detach().cpu()),
    }

  def train_step(self, data):
    self.optimizer.zero_grad()

    x = data[0].to(device=self.device, dtype=torch.float32)
    y = data[1].to(device=self.device)
    teacher_logits = None
    if len(data) > 3:
      teacher_logits = data[3].to(device=self.device, dtype=torch.float32)

    clip_logits, framewise_logits = self.forward(x, return_framewise=True)
    loss, metrics = self.compute_sed_losses(clip_logits, framewise_logits, y, teacher_logits)
    loss.backward()

    clip_norm = self.distillation_cfg().get("gradient_clip_norm")
    if clip_norm is not None:
      torch.nn.utils.clip_grad_norm_(self.parameters(), float(clip_norm))

    self.optimizer.step()
    self.last_train_step_metrics = metrics
    return loss.item()

  def validation_step(self, data):
    with torch.no_grad():
      x = data[0].to(device=self.device, dtype=torch.float32)
      y = data[1].to(device=self.device)
      teacher_logits = None
      if len(data) > 3:
        teacher_logits = data[3].to(device=self.device, dtype=torch.float32)

      clip_logits, framewise_logits = self.forward(x, return_framewise=True)
      loss, metrics = self.compute_sed_losses(clip_logits, framewise_logits, y, teacher_logits)
      self.last_validation_step_metrics = metrics
      y_hat = self.prediction_post_processing(clip_logits)

    return y_hat, loss.item()
