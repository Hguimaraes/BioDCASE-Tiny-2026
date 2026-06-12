# --
# Train the slim EfficientNet-B3 student (Perch-shaped) with Perch distillation,
# optionally with LAYER-TO-LAYER feature distillation onto Perch's spatial map.
#
# One loop, two experiments via --feature-distill:
#   off : logit distillation only  (alpha*KL(teacher_logits) + (1-a)*CE)
#   on  : logit distillation + a feature-hint loss between the student's pre-pool
#         B3 map and Perch's `spatial_embedding`. Spatial sizes match by design
#         (Perch-resolution front-end); the student's channels are projected up
#         to the teacher's with a 1x1 conv, then compared with L1 + cosine
#         (the BioME hint loss):  rec = L1(s, t);  sim = -logsigmoid(cos(s,t))
#
# Inputs (all keyed by wav stem):
#   - PCEN-Perch features  : output/02_features/cache_pcen_perch  (built by the pipeline)
#   - teacher logits       : experiments/perch/embeddings/perch_v2_cpu/teacher_mlp
#   - teacher spatial maps  : experiments/perch/embeddings/perch_v2_cpu/spatial   (export_spatial.py)
#
#   .venv/bin/python experiments/timm/train_effnetb3_l2l.py --feature-distill off
#   .venv/bin/python experiments/timm/train_effnetb3_l2l.py --feature-distill on --feature-weight 1.0

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

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen_perch'
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'
MODEL, CM, DM = 'efficientnet_b3', 0.13, 1.4   # same slim B3 as EffNetB3Slim


def load_split(split):
  files = sorted((CACHE / split).glob('*/*.npz'))
  xs, ys, stems = [], [], []
  for f in files:
    d = np.load(f)
    x = d['x'].astype(np.float32)
    xs.append(x.reshape(1, 128, -1))        # (1,128,T) Perch-res PCEN
    ys.append(f.parent.name)
    stems.append(f.stem)
  return np.stack(xs), np.array(ys), stems


def load_teacher_spatial(split):
  d = np.load(TEACHER / 'spatial' / '{}.npz'.format(split), allow_pickle=True)
  return {str(s): d['spatial'][i] for i, s in enumerate(d['stems'])}


class SlimB3Student(nn.Module):
  def __init__(self, n_classes, head_dim=64, dropout=0.2):
    super().__init__()
    from timm.models.efficientnet import _gen_efficientnet
    self.backbone = _gen_efficientnet(MODEL, channel_multiplier=CM, depth_multiplier=DM,
                                      in_chans=1, num_classes=0)
    emb = self.backbone.num_features
    self.feat_ch = emb
    self.classifier = nn.Sequential(
      nn.LayerNorm(emb), nn.Dropout(dropout),
      nn.Linear(emb, head_dim), nn.ReLU(), nn.Linear(head_dim, n_classes))

  def forward(self, x):
    feat = self.backbone.forward_features(x)        # (B, C, H, W) pre-pool map
    emb = self.backbone.forward_head(feat)          # global-pool -> (B, C)
    return self.classifier(emb), feat


def align_teacher(t, target_hw):
  """teacher spatial (B,Ct,Ht,Wt) -> (B,Ct,*target_hw); transpose if swapped, then resize."""
  Hs, Ws = target_hw
  if t.shape[-2:] == (Ws, Hs) and Hs != Ws:        # transposed (time<->freq) -> swap
    t = t.transpose(-1, -2)
  if t.shape[-2:] != (Hs, Ws):
    t = F.interpolate(t, size=(Hs, Ws), mode='bilinear', align_corners=False)
  return t


def feature_hint_loss(student_feat, teacher_feat, proj):
  s = proj(student_feat)                            # (B, Ct, Hs, Ws)
  t = align_teacher(teacher_feat, s.shape[-2:])
  rec = F.l1_loss(s, t)
  sim = -F.logsigmoid(F.cosine_similarity(s, t, dim=1)).mean()
  return rec + sim


def macro_auc(logits, y, n):
  from sklearn.metrics import roc_auc_score
  return roc_auc_score(np.eye(n)[y], torch.softmax(torch.from_numpy(logits), 1).numpy(),
                       average='macro', multi_class='ovr')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--feature-distill', choices=['on', 'off'], default='off')
  ap.add_argument('--feature-weight', type=float, default=1.0)
  ap.add_argument('--epochs', type=int, default=120)
  ap.add_argument('--batch', type=int, default=32)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'

  Xtr, ytr_n, str_stems = load_split('Train')
  Xva, yva_n, _ = load_split('Validation')
  classes = sorted(set(ytr_n) | set(yva_n)); c2i = {c: i for i, c in enumerate(classes)}
  ytr = np.array([c2i[c] for c in ytr_n]); yva = np.array([c2i[c] for c in yva_n])
  n = len(classes)
  print('Train {} Val {} | classes {} | device {} | feature_distill {}'.format(
      Xtr.shape, Xva.shape, n, dev, args.feature_distill))

  s2l = load_teacher_logits(str(TEACHER / 'teacher_mlp'), 'Train')
  Tlog = torch.tensor(np.stack([s2l[s] for s in str_stems]), dtype=torch.float32)

  model = SlimB3Student(n).to(dev)
  proj = None
  Tsp = None
  if args.feature_distill == 'on':
    s2s = load_teacher_spatial('Train')
    Tsp = np.stack([s2s[s] for s in str_stems]).astype(np.float32)   # (N,H,W,C) NHWC from TF
    # Perch (TF/Keras) spatial_embedding is channels-LAST: (16,4,1536)=(time,freq,ch).
    # -> NCHW for torch; align_teacher() then transposes (time,freq)->(freq,time) to
    # match the student's (freq,time) map.
    Tsp = np.ascontiguousarray(Tsp.transpose(0, 3, 1, 2))            # (N,C,H,W)=(N,1536,16,4)
    teacher_ch = Tsp.shape[1]
    proj = nn.Conv2d(model.feat_ch, teacher_ch, kernel_size=1).to(dev)
    print('  teacher spatial (NCHW) {} | projecting student {} -> {} ch'.format(
        Tsp.shape, model.feat_ch, teacher_ch))

  params = list(model.parameters()) + (list(proj.parameters()) if proj else [])
  opt = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

  Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr).long()
  Xva_t = torch.from_numpy(Xva)
  best = {'acc': 0.0, 'auc': float('nan'), 'epoch': 0}
  for epoch in range(args.epochs):
    model.train(); t0 = time.time(); perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), args.batch):
      idx = perm[i:i + args.batch]
      xb = Xtr_t[idx].to(dev); yb = ytr_t[idx].to(dev)
      opt.zero_grad()
      logits, feat = model(xb)
      hard = F.one_hot(yb, n).float()
      loss, _ = distillation_loss(logits, Tlog[idx].to(dev), hard,
                                  alpha=0.5, temperature=3.0, label_smoothing=0.1)
      if proj is not None:
        tf = torch.from_numpy(Tsp[idx.numpy()]).to(dev)
        loss = loss + args.feature_weight * feature_hint_loss(feat, tf, proj)
      loss.backward(); opt.step()
    sched.step()
    model.eval(); vl = []
    with torch.no_grad():
      for i in range(0, len(Xva_t), args.batch):
        vl.append(model(Xva_t[i:i + args.batch].to(dev))[0].cpu())
    vl = torch.cat(vl).numpy()
    acc = float((vl.argmax(1) == yva).mean()); auc = macro_auc(vl, yva, n)
    if acc > best['acc']: best = {'acc': acc, 'auc': auc, 'epoch': epoch + 1}
    print('  epoch {:3d}/{}  val acc {:.4f}  auc {:.4f}  ({:.0f}s)'.format(
        epoch + 1, args.epochs, acc, auc, time.time() - t0))

  print('\n=== slim-B3 (distill, feature_distill={}) ==='.format(args.feature_distill))
  print('  best val acc {:.4f}  auc {:.4f}  (epoch {})'.format(best['acc'], best['auc'], best['epoch']))
  print('  reference: PCEN-Baseline 0.6497 / 0.9296')

  import yaml, datetime
  out = Path(__file__).parent / 'results'; out.mkdir(parents=True, exist_ok=True)
  yaml.safe_dump({
    'model': 'effnetb3_slim_perch', 'feature_distill': args.feature_distill,
    'feature_weight': args.feature_weight, 'epochs': args.epochs, 'lr': args.lr,
    'seed': args.seed, 'val_acc': round(best['acc'], 4), 'val_auc': round(best['auc'], 4),
    'best_epoch': best['epoch'], 'finished_utc': datetime.datetime.utcnow().isoformat(),
  }, open(out / 'effnetb3_l2l_{}_s{}.yaml'.format(args.feature_distill, args.seed), 'w'), sort_keys=False)


if __name__ == '__main__':
  main()
