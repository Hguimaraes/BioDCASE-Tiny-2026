# --
# Phase B: distill the (original) Perch teacher into the student on the EXPANDED
# set = original 2.2k soundscape clips + agreement-filtered extra clips (XC birds
# + TAU background), all teacher-labeled. The real test of whether the data lifts
# the 0.6952 student.
#
# Reuses the proven D2 training loop (run_model_training: logit + embedding distill
# + EMA). We just assemble a combined dataset yielding the TeacherLogitDataset
# tuple (x, y, sid, teacher_logits, teacher_embedding):
#   - features: cache_pcen (original) + cache_pcen_extra (extra, from featurize_extra.py)
#   - teacher logits: original soft_logits_Train.npz + extra predictions.npz (kept)
#   - teacher embeddings: original Train.npz + extra clips.npz (kept), standardized
#
#   .venv/bin/python experiments/data/train_student_phaseB.py \
#       --extra-per-class 3000 --epochs 60
#
# NOTE: training a conv net on tens of thousands of clips is heavy on CPU; use
# --extra-per-class to bound it, or run on a GPU node. Eval is the ORIGINAL val.

import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.model_tiny_ml import Baseline
from pipeline_pytorch.model_training import run_model_training, run_validation_epoch
from pipeline_pytorch.distillation import load_teacher_logits
from pipeline_pytorch.paths import MODELS_DIR

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'
EXTRA_CACHE = ROOT / 'output' / '02_features' / 'cache_pcen_extra'
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'
XC = Path('/home/hguimaraes/datasets/extra/xc')
BG = Path('/home/hguimaraes/datasets/extra/background')


def logits_from_pred(npz):
  d = np.load(npz, allow_pickle=True)
  return {str(s): d['logits'][i] for i, s in enumerate(d['stems'])}


def emb_from_npz(npz):
  d = np.load(npz, allow_pickle=True)
  return {str(s): d['embeddings'][i].astype(np.float32) for i, s in enumerate(d['stems'])}


class TupleDS(torch.utils.data.Dataset):
  """yields (x, y, sid[, t_logits, t_emb]); with teacher -> matches the D2 embed path."""
  def __init__(self, X, y, Tlog=None, Temb=None):
    self.X, self.y, self.Tlog, self.Temb = X, y, Tlog, Temb
  def __len__(self): return len(self.y)
  def __getitem__(self, i):
    x = torch.from_numpy(self.X[i]); yy = torch.tensor(int(self.y[i])); sid = torch.tensor(i)
    if self.Tlog is None:
      return x, yy, sid
    return x, yy, sid, torch.from_numpy(self.Tlog[i]), torch.from_numpy(self.Temb[i])


def gather(split_glob, classes_idx, logit_d, emb_d, cap=0, seed=42, is_extra=False):
  """collect (npz_path, class_idx, stem) that have teacher logits+emb; cap extra per class."""
  rng = np.random.default_rng(seed)
  by_cls = {}
  for f in sorted(Path(p) for p in glob.glob(split_glob)):
    cls = f.parent.name
    if cls not in classes_idx or f.stem not in logit_d or f.stem not in emb_d:
      continue
    by_cls.setdefault(cls, []).append(f)
  items = []
  for cls, fs in by_cls.items():
    if is_extra and cap and len(fs) > cap:
      fs = [fs[i] for i in rng.choice(len(fs), cap, replace=False)]
    items += [(f, classes_idx[cls], f.stem) for f in fs]
  return items


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--extra-per-class', type=int, default=3000, help='cap extra clips per class (0 = all)')
  ap.add_argument('--epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=64)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)

  # canonical 11-class order = teacher head label_names
  ck = torch.load(TEACHER / 'teacher_mlp' / 'teacher_head.pt', map_location='cpu', weights_only=False)
  classes = [str(n) for n in ck['label_names']]; cidx = {c: i for i, c in enumerate(classes)}; n_cls = len(classes)

  # teacher logits + embeddings by stem (original + extra)
  logit_d = load_teacher_logits(str(TEACHER / 'teacher_mlp'), 'Train')
  logit_d.update(logits_from_pred(XC / 'predictions.npz'))
  logit_d.update(logits_from_pred(BG / 'predictions.npz'))
  emb_d = emb_from_npz(TEACHER / 'Train.npz')
  emb_d.update(emb_from_npz(XC / 'embeddings' / 'perch_v2_cpu' / 'clips.npz'))
  emb_d.update(emb_from_npz(BG / 'embeddings' / 'perch_v2_cpu' / 'clips.npz'))

  # train items: all original + capped extra (extra restricted to kept manifest via cache existing)
  orig = gather(str(CACHE / 'Train' / '*' / '*.npz'), cidx, logit_d, emb_d)
  extra = gather(str(EXTRA_CACHE / '*' / '*.npz'), cidx, logit_d, emb_d, cap=args.extra_per_class, seed=args.seed, is_extra=True)
  items = orig + extra
  print('train: {} original + {} extra = {} (cap {}/class) | classes {}'.format(
      len(orig), len(extra), len(items), args.extra_per_class or 'all', n_cls))

  # materialize arrays
  N = len(items)
  X = np.empty((N, 1, 40, 133), np.float32); y = np.empty(N, np.int64)
  Tlog = np.empty((N, n_cls), np.float32); Temb = np.empty((N, 1536), np.float32)
  for i, (f, ci, stem) in enumerate(items):
    X[i] = np.load(f)['x'].reshape(1, 40, 133); y[i] = ci
    Tlog[i] = logit_d[stem]; Temb[i] = emb_d[stem]
  mu, sd = Temb.mean(0, keepdims=True), Temb.std(0, keepdims=True) + 1e-6   # standardize emb (as D2)
  Temb = (Temb - mu) / sd

  # validation (original soundscape val; no teacher needed)
  vitems = [(f, cidx[f.parent.name]) for f in sorted((CACHE / 'Validation').glob('*/*.npz')) if f.parent.name in cidx]
  Xv = np.stack([np.load(f)['x'].reshape(1, 40, 133) for f, _ in vitems]).astype(np.float32)
  yv = np.array([c for _, c in vitems], np.int64)
  print('val: {}'.format(len(yv)))

  dl_tr = torch.utils.data.DataLoader(TupleDS(X, y, Tlog, Temb), batch_size=args.batch, shuffle=True)
  dl_va = torch.utils.data.DataLoader(TupleDS(Xv, yv), batch_size=128, shuffle=False)

  model = Baseline(input_shape=[1, 40, 133], num_classes=n_cls, save_path=str(MODELS_DIR),
                   device={'use_cpu': not torch.cuda.is_available(), 'device_name': 'cuda:0'},
                   criterion={'module': 'torch.nn', 'attr': 'CrossEntropyLoss', 'kwargs': {'label_smoothing': 0.0}},
                   optimizer={'module': 'torch.optim', 'attr': 'Adam', 'kwargs': {'lr': 0.001, 'betas': [0.9, 0.999]}},
                   verbose=False)

  cfg = {'model_training': {'num_epochs': args.epochs}, 'training_recipe': {
    'augmentation': {'enabled': False}, 'mixup': {'alpha': 0.0, 'p': 0.0},
    'scheduler': {'name': 'cosine', 'warmup_epochs': 5, 'min_lr_factor': 0.05},
    'best_checkpoint_metric': 'val_auc', 'early_stopping_patience': 0,
    'distillation': {'enabled': True, 'alpha': 0.5, 'temperature': 3.0, 'label_smoothing': 0.1,
                     'embed': {'enabled': True, 'weight': 1.0, 'mse_weight': 1.0, 'cos_weight': 1.0}},
    'ema': {'enabled': True, 'decay': 0.999}, 'class_balance': {'mode': 'none'}}}

  class_counts = np.bincount(y, minlength=n_cls)
  run_model_training(cfg, model, dl_tr, dl_va, label_dict=cidx, run_logger=None, class_counts=class_counts)

  acc, auc = (lambda r: (r[1], r[2]))(run_validation_epoch(model, dl_va))
  print('\n=== Phase B student (PCEN + logit + embed distill + EMA, +extra data) ===')
  print('  val acc {:.4f} auc {:.4f}  (D2 baseline 0.6952 / 0.9402)'.format(acc, auc))


if __name__ == '__main__':
  main()
