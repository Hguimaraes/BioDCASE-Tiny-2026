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
# MEMORY: the teacher .npz files are deflate-compressed (so they can't be mmap'd,
# and every NpzFile read fully decompresses). To stay light we (1) read only the
# small `stems` arrays to decide which clips to keep, (2) apply --extra-per-class
# BEFORE pulling any embeddings, then decompress each big array exactly once and
# keep only the rows we train on, and (3) load feature npz lazily per sample
# instead of pre-stacking one giant array. Eval is the ORIGINAL val.

import sys
import glob
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.model_tiny_ml import Baseline
from pipeline_pytorch.model_training import run_model_training, run_validation_epoch
from pipeline_pytorch.paths import MODELS_DIR

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'
EXTRA_CACHE = ROOT / 'output' / '02_features' / 'cache_pcen_extra'
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'
XC_DEFAULT = '/home/hguimaraes/datasets/extra/xc'
BG_DEFAULT = '/home/hguimaraes/datasets/extra/background'


def build_index(sources):
  """stem -> (source_idx, row). Reads ONLY the small `stems` array from each npz,
  so we can decide membership without decompressing the big embedding/logit array."""
  idx = {}
  for si, p in enumerate(sources):
    for r, s in enumerate(np.load(p)['stems']):
      idx[str(s)] = (si, r)
  return idx


def gather_rows(sources, idx, stems, key, dim):
  """Build (len(stems), dim) by pulling only the needed rows. Each big source array
  is decompressed exactly once, fancy-indexed for its rows, then freed."""
  out = np.empty((len(stems), dim), np.float32)
  by_src = defaultdict(list)
  for di, s in enumerate(stems):
    si, r = idx[s]
    by_src[si].append((di, r))
  for si, pairs in by_src.items():
    arr = np.load(sources[si])[key]                       # single decompress
    dst = np.fromiter((d for d, _ in pairs), int, len(pairs))
    src = np.fromiter((r for _, r in pairs), int, len(pairs))
    out[dst] = np.asarray(arr[src], np.float32)
    del arr
  return out


class LazyDS(torch.utils.data.Dataset):
  """Loads each feature npz on access (no giant pre-stacked array). With teacher
  tensors -> yields the D2 embed tuple; without -> (x, y, sid)."""
  def __init__(self, paths, y, Tlog=None, Temb=None):
    self.paths, self.y, self.Tlog, self.Temb = paths, y, Tlog, Temb
  def __len__(self): return len(self.y)
  def __getitem__(self, i):
    x = torch.from_numpy(np.load(self.paths[i])['x'].reshape(1, 40, 133).astype(np.float32))
    yy = torch.tensor(int(self.y[i])); sid = torch.tensor(i)
    if self.Tlog is None:
      return x, yy, sid
    return x, yy, sid, torch.from_numpy(self.Tlog[i]), torch.from_numpy(self.Temb[i])


def gather(split_glob, classes_idx, have, cap=0, seed=42, is_extra=False):
  """collect (npz_path, class_idx, stem) that have teacher logits+emb; cap extra per class."""
  rng = np.random.default_rng(seed)
  by_cls = {}
  for f in sorted(Path(p) for p in glob.glob(split_glob)):
    cls = f.parent.name
    if cls not in classes_idx or not have(f.stem):
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
  ap.add_argument('--extra-per-class', type=int, default=3000,
                  help='extra clips per class: 0 = none (original-only D2 repro), <0 = all (uncapped), N>0 = cap N/class')
  ap.add_argument('--epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=64)
  ap.add_argument('--num-workers', type=int, default=4, help='DataLoader workers for lazy feature IO')
  ap.add_argument('--xc', type=Path, default=XC_DEFAULT, help='Xeno-canto extra-data root (predictions.npz + embeddings/perch_v2_cpu/clips.npz)')
  ap.add_argument('--bg', type=Path, default=BG_DEFAULT, help='background extra-data root (predictions.npz + embeddings/perch_v2_cpu/clips.npz)')
  ap.add_argument('--lr', type=float, default=1e-3, help='base (peak) learning rate')
  ap.add_argument('--min-lr', type=float, default=1e-4, help='cosine floor LR at the final epoch')
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)

  # canonical 11-class order = teacher head label_names
  ck = torch.load(TEACHER / 'teacher_mlp' / 'teacher_head.pt', map_location='cpu', weights_only=False)
  classes = [str(n) for n in ck['label_names']]; cidx = {c: i for i, c in enumerate(classes)}; n_cls = len(classes)

  # teacher source files (read lazily: only `stems` now, big arrays later, once each).
  # extra (XC/BG) sources are added ONLY when extra data is requested, so 0 = none
  # is a clean original-only D2 reproduction that doesn't even touch the XC/BG files.
  use_extra = args.extra_per_class != 0
  EMB_SOURCES = [TEACHER / 'Train.npz']
  LOGIT_SOURCES = [TEACHER / 'teacher_mlp' / 'soft_logits_Train.npz']
  if use_extra:
    EMB_SOURCES += [args.xc / 'embeddings' / 'perch_v2_cpu' / 'clips.npz',
                    args.bg / 'embeddings' / 'perch_v2_cpu' / 'clips.npz']
    LOGIT_SOURCES += [args.xc / 'predictions.npz', args.bg / 'predictions.npz']
  emb_idx = build_index(EMB_SOURCES)
  logit_idx = build_index(LOGIT_SOURCES)
  have = lambda s: s in emb_idx and s in logit_idx

  # train items: all original + extra (cap applied BEFORE loading tensors).
  # extra_per_class: 0 -> none; <0 -> all (cap 0 disables capping); N>0 -> cap N/class
  orig = gather(str(CACHE / 'Train' / '*' / '*.npz'), cidx, have)
  if use_extra:
    cap = args.extra_per_class if args.extra_per_class > 0 else 0
    extra = gather(str(EXTRA_CACHE / '*' / '*.npz'), cidx, have, cap=cap, seed=args.seed, is_extra=True)
  else:
    extra = []
  items = orig + extra
  cap_label = 'none' if args.extra_per_class == 0 else ('all' if args.extra_per_class < 0 else args.extra_per_class)
  print('train: {} original + {} extra = {} (extra/class {}) | classes {}'.format(
      len(orig), len(extra), len(items), cap_label, n_cls))

  # pull only the teacher rows we actually train on
  stems = [s for _, _, s in items]
  paths = [str(f) for f, _, _ in items]
  y = np.array([ci for _, ci, _ in items], np.int64)
  Tlog = gather_rows(LOGIT_SOURCES, logit_idx, stems, 'logits', n_cls)
  Temb = gather_rows(EMB_SOURCES, emb_idx, stems, 'embeddings', 1536)
  mu, sd = Temb.mean(0, keepdims=True), Temb.std(0, keepdims=True) + 1e-6   # standardize emb (as D2)
  Temb = (Temb - mu) / sd

  # validation (original soundscape val; no teacher needed; loaded lazily too)
  vitems = [(str(f), cidx[f.parent.name]) for f in sorted((CACHE / 'Validation').glob('*/*.npz')) if f.parent.name in cidx]
  vpaths = [p for p, _ in vitems]; yv = np.array([c for _, c in vitems], np.int64)
  print('val: {}'.format(len(yv)))

  dl_tr = torch.utils.data.DataLoader(LazyDS(paths, y, Tlog, Temb), batch_size=args.batch,
                                      shuffle=True, num_workers=args.num_workers)
  dl_va = torch.utils.data.DataLoader(LazyDS(vpaths, yv), batch_size=128,
                                      shuffle=False, num_workers=args.num_workers)

  model = Baseline(input_shape=[1, 40, 133], num_classes=n_cls, save_path=str(MODELS_DIR),
                   device={'use_cpu': not torch.cuda.is_available(), 'device_name': 'cuda:0'},
                   criterion={'module': 'torch.nn', 'attr': 'CrossEntropyLoss', 'kwargs': {'label_smoothing': 0.0}},
                   optimizer={'module': 'torch.optim', 'attr': 'Adam', 'kwargs': {'lr': args.lr, 'betas': [0.9, 0.999]}},
                   verbose=False)

  # scheduler floor is expressed as a fraction of the base LR (min_lr_factor)
  min_lr_factor = min(args.min_lr / args.lr, 1.0)
  cfg = {'model_training': {'num_epochs': args.epochs}, 'training_recipe': {
    'augmentation': {'enabled': False}, 'mixup': {'alpha': 0.0, 'p': 0.0},
    'scheduler': {'name': 'cosine', 'warmup_epochs': 5, 'min_lr_factor': min_lr_factor},
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
