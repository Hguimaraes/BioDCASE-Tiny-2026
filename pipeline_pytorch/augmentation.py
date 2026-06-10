# --
# feature-space dynamic augmentation (track A)
#
# augmentations operate on the cached, min-max normalized [0, 1] mel
# features of shape (channel, mel, time) - e.g. (1, 40, 133) for the
# baseline config - so the on-device feature extraction stays untouched
# and fully deployable. applied per sample on the train split only,
# freshly sampled every epoch.

import sys
import torch
import numpy as np

from pathlib import Path

# add root path of project if called as main
if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]


class FeatureAugmentDataset(torch.utils.data.Dataset):
  """
  wraps a feature dataset and applies dynamic spectrogram augmentation
  """

  def __init__(self, dataset, cfg={}):

    # super constructor
    super().__init__()

    # members
    self.dataset = dataset

    # default config
    cfg_default = {
      'enabled': True,
      'time_mask': {'num': 2, 'max_width': 24, 'p': 0.5},
      'freq_mask': {'num': 2, 'max_width': 8, 'p': 0.5},
      'gaussian_noise': {'std': 0.03, 'p': 0.3},
      'mask_fill_value': 0.0,
    }

    # config update (shallow per augmentation entry)
    self.cfg = {**cfg_default, **cfg}


  def __len__(self):
    return len(self.dataset)


  def __getitem__(self, idx):

    # get original sample
    x, y, sid = self.dataset[idx]

    # skip
    if not self.cfg['enabled']: return x, y, sid

    # work on a copy, never mutate the cached features
    x = x.clone().to(dtype=torch.float32)

    # time masking - x shape: (channel, mel, time), time is the last axis
    c = self.cfg['time_mask']
    for _ in range(c['num']):
      if torch.rand(1).item() < c['p'] and x.shape[2] > c['max_width']:
        w = int(torch.randint(1, c['max_width'] + 1, (1,)).item())
        t0 = int(torch.randint(0, x.shape[2] - w, (1,)).item())
        x[:, :, t0:t0 + w] = self.cfg['mask_fill_value']

    # frequency masking - mel axis
    c = self.cfg['freq_mask']
    for _ in range(c['num']):
      if torch.rand(1).item() < c['p'] and x.shape[1] > c['max_width']:
        w = int(torch.randint(1, c['max_width'] + 1, (1,)).item())
        f0 = int(torch.randint(0, x.shape[1] - w, (1,)).item())
        x[:, f0:f0 + w, :] = self.cfg['mask_fill_value']

    # gaussian noise (features are [0, 1] normalized)
    c = self.cfg['gaussian_noise']
    if torch.rand(1).item() < c['p']:
      x = torch.clamp(x + torch.randn_like(x) * c['std'], 0.0, 1.0)

    return x, y, sid


def mixup_batch(x, y, num_classes, alpha=0.2, p=0.5):
  """
  batch-level mixup: returns mixed inputs and soft targets.
  y comes in as integer class labels, always returns (x, y_soft) where
  y_soft is a (batch, num_classes) probability matrix so the training
  step handles mixed and unmixed batches identically.
  """

  # one-hot soft targets
  y_soft = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=num_classes).to(torch.float32)

  # skip
  if alpha <= 0 or torch.rand(1).item() >= p: return x, y_soft

  # mixing coefficient and permutation
  lam = float(np.random.beta(alpha, alpha))
  perm = torch.randperm(x.shape[0])

  # mix
  x = lam * x + (1.0 - lam) * x[perm]
  y_soft = lam * y_soft + (1.0 - lam) * y_soft[perm]

  return x, y_soft


if __name__ == '__main__':
  """
  augmentation smoke test
  """

  # fake dataset
  class _Toy(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, idx): return torch.rand(1, 40, 133), torch.tensor(idx % 3), torch.tensor(idx)

  ds = FeatureAugmentDataset(_Toy())
  x, y, sid = ds[0]
  print('augmented sample:', x.shape, x.min().item(), x.max().item(), y.item())

  xb = torch.stack([ds[i][0] for i in range(8)])
  yb = torch.stack([ds[i][1] for i in range(8)])
  xm, ym = mixup_batch(xb, yb, num_classes=3, alpha=0.2, p=1.0)
  print('mixup batch:', xm.shape, ym.shape, 'row sums:', ym.sum(dim=1).tolist())
