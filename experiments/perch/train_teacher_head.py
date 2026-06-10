# --
# Perch teacher head training + soft-label export
#
# Trains an 11-class head on top of frozen Perch v2 embeddings (exported by
# export_embeddings.py), evaluates it on the validation split (this is the
# teacher's own accuracy = a reference ceiling for the distilled student),
# and exports the teacher's logits for every clip so the student can distill
# from them, keyed by wav stem for cross-environment alignment.
#
# Runs in either venv (pure torch + numpy). Example:
#   .venv/bin/python experiments/perch/train_teacher_head.py \
#       --emb-dir experiments/perch/embeddings/perch_v2_cpu \
#       --head mlp --epochs 200
#
# Outputs under <emb-dir>/teacher_<head>/:
#   teacher_head.pt, metrics.yaml, soft_logits_<split>.npz (stems, logits, labels)

import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn

from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score


def parse_args():
  p = argparse.ArgumentParser(description='Train a classifier head on Perch embeddings.')
  p.add_argument('--emb-dir', required=True, help='dir with <split>.npz embedding files')
  p.add_argument('--train-split', default='Train')
  p.add_argument('--val-split', default='Validation')
  p.add_argument('--head', choices=['linear', 'mlp'], default='mlp')
  p.add_argument('--hidden', type=int, default=512)
  p.add_argument('--dropout', type=float, default=0.3)
  p.add_argument('--epochs', type=int, default=200)
  p.add_argument('--lr', type=float, default=1e-3)
  p.add_argument('--weight-decay', type=float, default=1e-4)
  p.add_argument('--batch-size', type=int, default=128)
  p.add_argument('--temperature', type=float, default=2.0, help='softmax temperature for exported soft logits info only')
  p.add_argument('--seed', type=int, default=42)
  p.add_argument('--standardize', action='store_true', default=True, help='z-score embeddings using train stats')
  return p.parse_args()


def load_split(emb_dir, split):
  d = np.load(Path(emb_dir) / '{}.npz'.format(split), allow_pickle=True)
  return d['stems'], d['embeddings'].astype(np.float32), d['labels'].astype(np.int64), [str(n) for n in d['label_names']]


def build_head(kind, in_dim, num_classes, hidden, dropout):
  if kind == 'linear':
    return nn.Linear(in_dim, num_classes)
  return nn.Sequential(
    nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
    nn.Linear(hidden, num_classes),
  )


def main():
  args = parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # data
  tr_stems, Xtr, ytr, names = load_split(args.emb_dir, args.train_split)
  va_stems, Xva, yva, _ = load_split(args.emb_dir, args.val_split)
  num_classes = len(names)
  print('Teacher head: {} | train {} | val {} | dim {} | classes {}'.format(args.head, len(ytr), len(yva), Xtr.shape[1], num_classes))

  # standardize with train stats
  mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
  if args.standardize:
    Xtr_n, Xva_n = (Xtr - mu) / sd, (Xva - mu) / sd
  else:
    Xtr_n, Xva_n = Xtr, Xva

  # tensors
  Xtr_t = torch.from_numpy(Xtr_n).to(device); ytr_t = torch.from_numpy(ytr).to(device)
  Xva_t = torch.from_numpy(Xva_n).to(device)

  # class-balanced loss (background is over-represented)
  counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
  weights = torch.from_numpy((counts.sum() / (counts + 1e-6)) / num_classes).float().to(device)

  # model
  head = build_head(args.head, Xtr.shape[1], num_classes, args.hidden, args.dropout).to(device)
  opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
  crit = nn.CrossEntropyLoss(weight=weights)

  best = {'auc': -1, 'state': None, 'epoch': -1, 'acc': -1}
  n = len(ytr)

  for epoch in range(args.epochs):
    head.train()
    perm = torch.randperm(n, device=device)
    for i in range(0, n, args.batch_size):
      idx = perm[i:i + args.batch_size]
      opt.zero_grad()
      loss = crit(head(Xtr_t[idx]), ytr_t[idx])
      loss.backward(); opt.step()
    sched.step()

    # validation
    head.eval()
    with torch.no_grad():
      logits_va = head(Xva_t).cpu().numpy()
    prob_va = softmax(logits_va, axis=1)
    acc = accuracy_score(yva, logits_va.argmax(1))
    try: auc = roc_auc_score(yva, prob_va, multi_class='ovr', average='macro')
    except ValueError: auc = float('nan')

    if auc > best['auc']:
      best = {'auc': float(auc), 'acc': float(acc), 'epoch': epoch + 1, 'state': {k: v.cpu().clone() for k, v in head.state_dict().items()}}

    if (epoch + 1) % 20 == 0 or epoch == 0:
      print('epoch {:03d} | val acc {:.4f} auc {:.4f} (best auc {:.4f} @ {})'.format(epoch + 1, acc, auc, best['auc'], best['epoch']))

  print('\nBest teacher head: val acc {:.4f} | val auc {:.4f} (epoch {})'.format(best['acc'], best['auc'], best['epoch']))

  # restore best and export
  head.load_state_dict(best['state'])
  out_dir = Path(args.emb_dir) / 'teacher_{}'.format(args.head)
  out_dir.mkdir(parents=True, exist_ok=True)

  torch.save({'state_dict': head.state_dict(), 'mu': mu, 'sd': sd, 'head': args.head,
              'hidden': args.hidden, 'dropout': args.dropout, 'label_names': names,
              'standardize': args.standardize}, out_dir / 'teacher_head.pt')

  yaml.safe_dump({'head': args.head, 'val_acc': best['acc'], 'val_auc': best['auc'],
                  'best_epoch': best['epoch'], 'num_classes': num_classes,
                  'temperature': args.temperature, 'label_names': names},
                 open(out_dir / 'metrics.yaml', 'w'), sort_keys=False)

  # export soft logits for all splits (raw logits; temperature applied at distill time)
  head.eval()
  for split, stems, X, y in [(args.train_split, tr_stems, Xtr_n, ytr), (args.val_split, va_stems, Xva_n, yva)]:
    with torch.no_grad():
      logits = head(torch.from_numpy(X).to(device)).cpu().numpy().astype(np.float32)
    np.savez_compressed(out_dir / 'soft_logits_{}.npz'.format(split), stems=stems, logits=logits, labels=y)
    print('exported soft logits: {} -> {}'.format(split, logits.shape))

  print('\nTeacher head + soft logits written under: {}'.format(out_dir))


if __name__ == '__main__':
  main()
