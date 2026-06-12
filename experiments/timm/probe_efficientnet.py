# --
# Small probe: timm test_efficientnet_ln (tiny ImageNet model) as a frozen
# feature extractor on our PCEN-mel spectrograms.
#
# Recipe (same shape as the Perch teacher): frozen backbone -> 256-d MEAN
# (global-avg-pooled) embedding -> small MLP head -> 11 logits. This answers a
# narrow question: do a tiny generic-vision model's features transfer to our
# bioacoustic PCEN spectrograms? It is NOT a serious teacher (43.9% ImageNet
# top-1, 0.4M params / 0.1 GMACs @160 -- ~4x our deploy budget) -- a probe only.
#
# Reuses the already-built PCEN cache (output/02_features/cache_pcen), so the
# input is exactly our best front-end. Embeddings are cached to disk so the
# MLP can be retrained cheaply.
#
#   .venv/bin/python experiments/timm/probe_efficientnet.py
#
# Frozen backbone runs on CPU; light, but will contend with any training job.

import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'
EMB_OUT = Path(__file__).parent / 'embeddings'
MODEL = 'test_efficientnet_ln.r160_in1k'
RES = 160
SEED = 1


def load_split(split):
  """glob cached PCEN npz for a split -> (X (N,1,40,133), labels (N,), classes)."""
  files = sorted((CACHE / split).glob('*/*.npz'))
  xs, ys = [], []
  for f in files:
    d = np.load(f)
    # cached features are stored FLATTENED (40*133=5320); restore (1,40,133)
    xs.append(d['x'].astype(np.float32).reshape(1, 40, 133))
    ys.append(f.parent.name)          # class = parent folder
  X = np.stack(xs)                     # (N,1,40,133)
  return X, np.array(ys)


def extract_embeddings(X, model, mean, std, device, batch=64):
  """PCEN (N,1,40,133) -> ImageNet-style (N,3,160,160) -> 256-d mean embedding."""
  embs = []
  model.eval()
  with torch.no_grad():
    for i in range(0, len(X), batch):
      xb = torch.from_numpy(X[i:i + batch]).to(device)           # (B,1,40,133)
      xb = xb.repeat(1, 3, 1, 1)                                  # -> 3 channels
      xb = F.interpolate(xb, size=(RES, RES), mode='bilinear', align_corners=False)
      xb = (xb - mean) / std                                      # ImageNet norm
      embs.append(model(xb).cpu())                                # (B,256) pooled mean embedding
  return torch.cat(embs).numpy()


class MLPHead(nn.Module):
  def __init__(self, in_dim, n_classes, hidden=128, p=0.3):
    super().__init__()
    self.net = nn.Sequential(
      nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
      nn.Linear(hidden, n_classes))

  def forward(self, x):
    return self.net(x)


def macro_auc(logits, y, n_classes):
  try:
    from sklearn.metrics import roc_auc_score
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return roc_auc_score(np.eye(n_classes)[y], prob, average='macro', multi_class='ovr')
  except Exception as e:
    print('  (auc skipped: {})'.format(e))
    return float('nan')


def main():
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  device = 'cpu'
  import timm

  print('Loading PCEN cache...')
  Xtr, ytr_names = load_split('Train')
  Xva, yva_names = load_split('Validation')
  classes = sorted(set(ytr_names) | set(yva_names))
  cls_to_idx = {c: i for i, c in enumerate(classes)}
  ytr = np.array([cls_to_idx[c] for c in ytr_names])
  yva = np.array([cls_to_idx[c] for c in yva_names])
  n_classes = len(classes)
  print('  Train {}  Val {}  classes {}'.format(Xtr.shape, Xva.shape, n_classes))

  # frozen backbone (num_classes=0 -> the pooled mean embedding)
  print('Loading {} ...'.format(MODEL))
  model = timm.create_model(MODEL, pretrained=True, num_classes=0).to(device)
  for p in model.parameters():
    p.requires_grad = False
  dcfg = timm.data.resolve_model_data_config(model)
  mean = torch.tensor(dcfg['mean']).view(1, 3, 1, 1)
  std = torch.tensor(dcfg['std']).view(1, 3, 1, 1)
  print('  data cfg: input {} mean {} std {}'.format(dcfg['input_size'], dcfg['mean'], dcfg['std']))

  EMB_OUT.mkdir(parents=True, exist_ok=True)
  cache = EMB_OUT / 'emb_{}.npz'.format(MODEL.split('.')[0])
  if cache.exists():
    print('Using cached embeddings: {}'.format(cache.name))
    z = np.load(cache)
    Etr, Eva = z['Etr'], z['Eva']
  else:
    t = time.time()
    print('Extracting embeddings (frozen backbone, CPU)...')
    Etr = extract_embeddings(Xtr, model, mean, std, device)
    Eva = extract_embeddings(Xva, model, mean, std, device)
    np.savez_compressed(cache, Etr=Etr, Eva=Eva)
    print('  done in {:.0f}s -> {} (dim {})'.format(time.time() - t, cache.name, Etr.shape[1]))

  # train MLP head on frozen embeddings
  Etr_t = torch.from_numpy(Etr).float(); ytr_t = torch.from_numpy(ytr).long()
  Eva_t = torch.from_numpy(Eva).float()
  head = MLPHead(Etr.shape[1], n_classes)
  opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
  lossf = nn.CrossEntropyLoss()

  best = {'acc': 0.0, 'auc': float('nan'), 'epoch': 0}
  print('Training MLP head...')
  for epoch in range(200):
    head.train()
    perm = torch.randperm(len(Etr_t))
    for i in range(0, len(perm), 64):
      idx = perm[i:i + 64]
      opt.zero_grad()
      loss = lossf(head(Etr_t[idx]), ytr_t[idx])
      loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
      logits = head(Eva_t).numpy()
    acc = float((logits.argmax(1) == yva).mean())
    if acc > best['acc']:
      best = {'acc': acc, 'auc': macro_auc(logits, yva, n_classes), 'epoch': epoch + 1}
  print('\n=== {} frozen-embedding + MLP probe ==='.format(MODEL))
  print('  best val acc = {:.4f}  macro auc = {:.4f}  (epoch {})'.format(best['acc'], best['auc'], best['epoch']))
  print('  reference: PCEN-Baseline (full distilled CNN) 0.6497 / 0.9296')


if __name__ == '__main__':
  main()
