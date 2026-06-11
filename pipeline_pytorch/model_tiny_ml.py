# --
# model tiny ml

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

# add root path of project if called as main
if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from pipeline_pytorch.model_base import ModelBase


class Baseline(ModelBase):
  """
  model tiny ml - overwrite model base
  """

  def define_network_structure(self, n_filters = 32, dropout=0.05):

    assert len(self.cfg['input_shape']) == 3

    # Feature extractor
    self.features = nn.Sequential(
        nn.Conv2d(self.cfg['input_shape'][0], n_filters, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(n_filters, n_filters * 2, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(4),

        nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
    )

    # Classifier
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(n_filters * 4, 32),
        nn.ReLU(),
        nn.Linear(32, self.cfg['num_classes'])
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


class DepthwiseSeparableBlock(nn.Module):
  """
  MobileNet-style depthwise-separable block: 3x3 depthwise conv (optionally
  strided) + 1x1 pointwise conv, each followed by BatchNorm + ReLU. All ops
  are int8 TFLite / esp-nn friendly (BN folds into the preceding conv).
  """

  def __init__(self, in_ch, out_ch, stride=1):
    super().__init__()
    self.block = nn.Sequential(
      # depthwise
      nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
      nn.BatchNorm2d(in_ch),
      nn.ReLU(),
      # pointwise
      nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
    )

  def forward(self, x):
    return self.block(x)


class SlimCNN(ModelBase):
  """
  Track B student: a depthwise-separable CNN on the baseline mel input
  (1 x mel x time). Replaces the baseline's dense 3x3 convs with cheaper
  DS blocks, freeing parameter budget for more depth/width. Architecture
  hyperparameters are read from config kwargs (defaults below) so width and
  depth can be swept.
  """

  def define_network_structure(self):

    assert len(self.cfg['input_shape']) == 3

    # arch hyperparameters (overridable via model kwargs in config)
    stem_ch = self.cfg.get('stem_ch', 24)
    block_widths = self.cfg.get('block_widths', [48, 64, 96, 128])
    block_strides = self.cfg.get('block_strides', [2, 2, 2, 1])
    head_dim = self.cfg.get('head_dim', 64)
    dropout = self.cfg.get('dropout', 0.1)
    assert len(block_widths) == len(block_strides), "block_widths and block_strides must match"

    # stem: standard 3x3 conv (cheap at 1 input channel)
    layers = [
      nn.Conv2d(self.cfg['input_shape'][0], stem_ch, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(stem_ch),
      nn.ReLU(),
    ]

    # depthwise-separable blocks
    in_ch = stem_ch
    for out_ch, stride in zip(block_widths, block_strides):
      layers.append(DepthwiseSeparableBlock(in_ch, out_ch, stride=stride))
      in_ch = out_ch

    # global average pooling
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    self.features = nn.Sequential(*layers)

    # classifier head
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(dropout),
      nn.Linear(in_ch, head_dim),
      nn.ReLU(),
      nn.Linear(head_dim, self.cfg['num_classes']),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x



if __name__ == '__main__':
  """
  model tiny ml
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'config.yaml'))

  # params
  num_classes = 11
  num_samples = 4

  # test data sample
  y = torch.randint(0, num_classes-1, (num_samples,))

  # model
  x = torch.randn(num_samples, 1, 133, 40)
  model = Baseline(cfg['pytorch_framework']['model'], input_shape=tuple(x.shape[1:]), num_classes=num_classes)
  model.info()

  # data structure
  data = (x, y)

  # to train mode
  model.set_model_to_training_mode()

  # train loop
  for epoch in range(50):

    # train model
    loss = model.train_step(data)
    print("Epoch {:03} with loss: {:6f}".format(epoch + 1, loss))

  # eval mode (must be done to disable for instance dropout)
  model.set_model_to_evaluation_mode()

  # validation step
  y_pred, loss = model.validation_step(data)

  print("actual: ", y.numpy())
  print("prediction: ", y_pred)
  print("loss: ", loss)
  print("acc: ", np.mean(y.numpy() == np.argmax(y_pred, axis=-1)))

  # save model
  model.save()
