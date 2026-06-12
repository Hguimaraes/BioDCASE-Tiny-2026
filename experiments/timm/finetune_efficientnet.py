# --
# End-to-end fine-tune of timm test_efficientnet_ln on our PCEN-mel spectrograms.
#
# Unlike probe_efficientnet.py (frozen backbone + MLP probe), here the ENTIRE
# network is trainable, initialized from ImageNet weights. PCEN (1,40,133) is
# adapted to the model's image input (3 ch, 160x160, ImageNet-normalized),
# mean-pooled to a 256-d embedding, then an MLP head -> 11 logits.
#
# Two modes, to compare against our best (PCEN-Baseline 0.6497 / 0.9296, which
# uses Perch distillation):
#   --distill off : plain cross-entropy fine-tune
#   --distill on  : Perch-logit distillation (alpha=0.5, T=3.0, ls=0.1) -- the
#                   same recipe/teacher as PCEN-Baseline, apples-to-apples.
#
# Caveat: 0.4M params / 0.1 GMACs @160 is ~4x our deploy budget -> research
# probe, not a deployable submission candidate.
#
#   .venv/bin/python experiments/timm/finetune_efficientnet.py --distill off
#   .venv/bin/python experiments/timm/finetune_efficientnet.py --distill on

import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.distillation import distillation_loss, load_teacher_logits

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'
TEACHER_DIR = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu' / 'teacher_mlp'
MODEL = 'test_efficientnet_ln.r160_in1k'
RES = 160


def load_split(split):
  files = sorted((CACHE / split).glob('*/*.npz'))
  xs, ys, stems = [], [], []
  for f in files:
    d = np.load(f)
    xs.append(d['x'].astype(np.float32).reshape(1, 40, 133))   # cache is flattened
    ys.append(f.parent.name)
    stems.append(f.stem)
  return np.stack(xs), np.array(ys), stems


class EffNetStudent(nn.Module):
  """ImageNet-init EfficientNet (trainable) + MLP head on its mean embedding."""

  def __init__(self, n_classes, mean, std, hidden=128, p=0.3):
    super().__init__()
    import timm
    self.backbone = timm.create_model(MODEL, pretrained=True, num_classes=0)  # -> pooled mean emb
    emb_dim = self.backbone.num_features
    self.head = nn.Sequential(
      nn.LayerNorm(emb_dim), nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(p),
      nn.Linear(hidden, n_classes))
    self.register_buffer('mean', mean)
    self.register_buffer('std', std)

  def forward(self, x):                       # x: (B,1,40,133) PCEN in [0,1]
    x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(RES, RES), mode='bilinear', align_corners=False)
    x = (x - self.mean) / self.std            # ImageNet normalization
    return self.head(self.backbone(x))        # mean embedding -> MLP -> logits


def macro_auc(logits, y, n_classes):
  from sklearn.metrics import roc_auc_score
  prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
  return roc_auc_score(np.eye(n_classes)[y], prob, average='macro', multi_class='ovr')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--distill', choices=['on', 'off'], default='off')
  ap.add_argument('--epochs', type=int, default=30)
  ap.add_argument('--batch', type=int, default=32)
  ap.add_argument('--lr', type=float, default=3e-4)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  Xtr, ytr_n, str_stems = load_split('Train')
  Xva, yva_n, _ = load_split('Validation')
  classes = sorted(set(ytr_n) | set(yva_n))
  c2i = {c: i for i, c in enumerate(classes)}
  ytr = np.array([c2i[c] for c in ytr_n]); yva = np.array([c2i[c] for c in yva_n])
  n_classes = len(classes)
  print('Train {} Val {} classes {} | device {} | distill {}'.format(
      Xtr.shape, Xva.shape, n_classes, device, args.distill))

  # teacher logits aligned by stem (distill mode)
  T = None
  if args.distill == 'on':
    s2l = load_teacher_logits(str(TEACHER_DIR), 'Train')
    miss = [s for s in str_stems if s not in s2l]
    assert not miss, 'missing teacher logits for {} stems e.g. {}'.format(len(miss), miss[:2])
    T = torch.tensor(np.stack([s2l[s] for s in str_stems]), dtype=torch.float32)
    print('  teacher logits aligned: {}'.format(tuple(T.shape)))

  import timm
  dcfg = timm.data.resolve_model_data_config(timm.create_model(MODEL, pretrained=True, num_classes=0))
  mean = torch.tensor(dcfg['mean']).view(1, 3, 1, 1)
  std = torch.tensor(dcfg['std']).view(1, 3, 1, 1)
  model = EffNetStudent(n_classes, mean, std).to(device)
  n_params = sum(p.numel() for p in model.parameters())
  print('  trainable params: {} (ALL unfrozen)'.format(n_params))

  Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr).long()
  Xva_t = torch.from_numpy(Xva); yva_t = torch.from_numpy(yva).long()
  opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

  best = {'acc': 0.0, 'auc': float('nan'), 'epoch': 0}
  for epoch in range(args.epochs):
    model.train(); t = time.time(); perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), args.batch):
      idx = perm[i:i + args.batch]
      xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device)
      opt.zero_grad()
      logits = model(xb)
      if args.distill == 'on':
        hard = F.one_hot(yb, n_classes).float()
        loss, _ = distillation_loss(logits, T[idx].to(device), hard,
                                    alpha=0.5, temperature=3.0, label_smoothing=0.1)
      else:
        loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
      loss.backward(); opt.step()
    sched.step()
    # validation
    model.eval(); vlogits = []
    with torch.no_grad():
      for i in range(0, len(Xva_t), args.batch):
        vlogits.append(model(Xva_t[i:i + args.batch].to(device)).cpu())
    vlogits = torch.cat(vlogits).numpy()
    acc = float((vlogits.argmax(1) == yva).mean())
    auc = macro_auc(vlogits, yva, n_classes)
    if acc > best['acc']:
      best = {'acc': acc, 'auc': auc, 'epoch': epoch + 1}
    print('  epoch {:3d}/{}  val acc {:.4f}  auc {:.4f}  ({:.0f}s)'.format(
        epoch + 1, args.epochs, acc, auc, time.time() - t))

  print('\n=== {} END-TO-END FINE-TUNE (distill={}) ==='.format(MODEL, args.distill))
  print('  best val acc {:.4f}  auc {:.4f}  (epoch {})'.format(best['acc'], best['auc'], best['epoch']))
  print('  reference: PCEN-Baseline (distilled CNN) 0.6497 / 0.9296  | frozen probe 0.5738 / 0.8859')

  # persist the result in the repo (not /tmp) so it survives like a run record
  import yaml, datetime
  out_dir = Path(__file__).parent / 'results'
  out_dir.mkdir(parents=True, exist_ok=True)
  out = out_dir / 'effnet_distill-{}_s{}.yaml'.format(args.distill, args.seed)
  yaml.safe_dump({
    'model': MODEL, 'mode': 'finetune_unfrozen', 'distill': args.distill,
    'input': 'pcen_mel', 'res': RES, 'epochs': args.epochs, 'batch': args.batch,
    'lr': args.lr, 'seed': args.seed, 'params': int(n_params),
    'val_acc': round(best['acc'], 4), 'val_auc': round(best['auc'], 4),
    'best_epoch': best['epoch'],
    'finished_utc': datetime.datetime.utcnow().isoformat(),
  }, open(out, 'w'), sort_keys=False)
  print('  saved -> {}'.format(out.relative_to(ROOT)))


if __name__ == '__main__':
  main()
