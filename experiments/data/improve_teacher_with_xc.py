# --
# Step 4 (Phase A): does adding agreement-filtered Xeno-canto clips improve the
# Perch teacher HEAD? (Perch encoder is frozen; only the 11-class head retrains.)
#
# Sweeps how much XC to add (capped per class) and gates on the ORIGINAL
# soundscape validation set -- the XC clips are focal, the val is soundscape, so
# more focal data can help (more exemplars/class) or hurt (domain shift). An
# original-only control reproduces the 0.8925 baseline. Reuses the exact head /
# training recipe from train_teacher_head.py for a fair comparison.
#
#   .venv/bin/python experiments/data/improve_teacher_with_xc.py \
#       --xc-emb /home/hguimaraes/datasets/extra/xc/embeddings/perch_v2_cpu/clips.npz \
#       --kept   /home/hguimaraes/datasets/extra/xc/kept_clips.txt \
#       --caps 1000 3000 10000 0          # 0 = use all agreed clips

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.perch.train_teacher_head import build_head, load_split

ORIG_DIR = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'


def train_eval(Xtr, ytr, Xva, yva, n_cls, epochs, hidden=512, dropout=0.3, lr=1e-3, wd=1e-4, bs=512, seed=42):
  """train the MLP head (train_teacher_head recipe) and return best val acc/auc."""
  torch.manual_seed(seed); np.random.seed(seed)
  mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6   # train-stat standardize
  Xtr_t = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32)); ytr_t = torch.from_numpy(ytr)
  Xva_t = torch.from_numpy(((Xva - mu) / sd).astype(np.float32))
  counts = np.bincount(ytr, minlength=n_cls).astype(np.float32)
  w = torch.from_numpy(((counts.sum() / (counts + 1e-6)) / n_cls).astype(np.float32))   # class-balanced
  head = build_head('mlp', Xtr.shape[1], n_cls, hidden, dropout)
  opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
  crit = nn.CrossEntropyLoss(weight=w)
  best = {'auc': -1.0, 'acc': -1.0, 'epoch': -1}
  n = len(ytr)
  for ep in range(epochs):
    head.train(); perm = torch.randperm(n)
    for i in range(0, n, bs):
      idx = perm[i:i + bs]; opt.zero_grad()
      crit(head(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
    sched.step()
    head.eval()
    with torch.no_grad(): lv = head(Xva_t).numpy()
    pred = lv.argmax(1)
    acc = accuracy_score(yva, pred)
    try: auc = roc_auc_score(yva, softmax(lv, 1), multi_class='ovr', average='macro')
    except ValueError: auc = float('nan')
    if auc > best['auc']: best = {'auc': float(auc), 'acc': float(acc), 'epoch': ep + 1, 'pred': pred.copy()}
  # per-class recall at the best-AUC epoch (to watch Background under imbalance)
  best['recall'] = np.array([(best['pred'][yva == c] == c).mean() if (yva == c).any() else np.nan
                             for c in range(n_cls)])
  return best


def load_agreed(emb_npz, kept_path, name_to_idx):
  """load an embeddings npz, subset to the kept-manifest stems, map folder names -> 11-class idx."""
  d = np.load(emb_npz, allow_pickle=True)
  stems = np.array([str(s) for s in d['stems']])
  cls_names = [str(n) for n in d['label_names']]
  kept = {Path(l.strip()).stem for l in open(kept_path) if l.strip()}
  m = np.array([s in kept for s in stems])
  idx = np.array([name_to_idx[cls_names[i]] for i in d['labels'][m]])
  return d['embeddings'].astype(np.float32)[m], idx


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--xc-emb', required=True, help='XC bird clips embeddings npz')
  ap.add_argument('--kept', required=True, help='XC kept_clips.txt from filter_clips select')
  ap.add_argument('--bg-emb', default=None, help='background clips embeddings npz (optional, adds Background)')
  ap.add_argument('--bg-kept', default=None, help='background kept_clips.txt (optional)')
  ap.add_argument('--caps', type=int, nargs='+', default=[1000, 3000, 10000, 0], help='max added clips PER CLASS (0 = all); applies to birds + background')
  ap.add_argument('--epochs', type=int, default=60)        # head converges fast (orig best ~epoch 5)
  ap.add_argument('--seed', type=int, default=42)
  args = ap.parse_args()

  # original 11-class embeddings (the label space)
  _, Xo, yo, names = load_split(str(ORIG_DIR), 'Train')
  _, Xva, yva, _ = load_split(str(ORIG_DIR), 'Validation')
  n_cls = len(names); name_to_idx = {n: i for i, n in enumerate(names)}
  Xo = Xo.astype(np.float32)

  # agreed extra clips (birds + optional background), mapped to the 11-class idx
  add_emb, add_idx = load_agreed(args.xc_emb, args.kept, name_to_idx)
  n_bird, n_bg = len(add_idx), 0
  if args.bg_emb and args.bg_kept:
    bg_emb, bg_idx = load_agreed(args.bg_emb, args.bg_kept, name_to_idx)
    n_bg = len(bg_idx)
    add_emb = np.concatenate([add_emb, bg_emb]); add_idx = np.concatenate([add_idx, bg_idx])
  print('original train {} | agreed extra {} ({} bird + {} bg) | val {} | classes {}'.format(
      len(yo), len(add_idx), n_bird, n_bg, len(yva), n_cls))

  bg_idx = name_to_idx.get('Background', -1)
  def rec_of(r, c): return r['recall'][c] if 0 <= c < n_cls else float('nan')

  rng = np.random.default_rng(args.seed)
  print('\n(bg_rec = Background recall; min_rec = worst class. Watch these for imbalance damage.)', flush=True)
  print('{:<22} {:>8} {:>7} {:>7} {:>7} {:>7}'.format('config', 'train_n', 'acc', 'auc', 'bg_rec', 'min_rec'), flush=True)
  print('-' * 62, flush=True)
  print('  ... training original-only control (n={}) ...'.format(len(yo)), flush=True)
  base = train_eval(Xo, yo, Xva, yva, n_cls, args.epochs, seed=args.seed)
  print('{:<22} {:>8} {:>7.4f} {:>7.4f} {:>7.3f} {:>7.3f}'.format(
      'original only (control)', len(yo), base['acc'], base['auc'], rec_of(base, bg_idx), np.nanmin(base['recall'])), flush=True)

  for cap in args.caps:
    # per-class cap on the added clips (birds + background alike)
    sel = []
    for c in range(n_cls):
      ci = np.where(add_idx == c)[0]
      if cap and len(ci) > cap: ci = rng.choice(ci, cap, replace=False)
      sel.append(ci)
    sel = np.concatenate(sel) if sel else np.array([], dtype=int)
    Xtr = np.concatenate([Xo, add_emb[sel]]); ytr = np.concatenate([yo, add_idx[sel]])
    tag = '+extra all/cls' if cap == 0 else '+extra <= {}/cls'.format(cap)
    print('  ... training {} (n={}) ...'.format(tag, len(ytr)), flush=True)
    r = train_eval(Xtr, ytr, Xva, yva, n_cls, args.epochs, seed=args.seed)
    flag = '  <-- beats control' if r['auc'] > base['auc'] else ''
    print('{:<22} {:>8} {:>7.4f} {:>7.4f} {:>7.3f} {:>7.3f}{}'.format(
        tag, len(ytr), r['acc'], r['auc'], rec_of(r, bg_idx), np.nanmin(r['recall']), flag), flush=True)

  print('\nbaseline teacher (saved): val_acc 0.8925 / val_auc 0.9925', flush=True)


if __name__ == '__main__':
  main()
