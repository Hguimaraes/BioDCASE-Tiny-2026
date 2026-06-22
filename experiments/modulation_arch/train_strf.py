# --
# Architecture probe: StrfBaseline -- the proven D2 `Baseline` tiny CNN with its
# first 3x3 conv swapped for a 2-D Gabor spectro-temporal receptive-field filterbank
# (GaborSTRFConv). The STRF is the one signal-processing bias that helped (best AUC
# at ~1x Baseline params); everything else (body, GAP descriptor, D2 recipe: PCEN +
# Perch logit KD + 1536-d embedding distillation + EMA) is unchanged. All standard
# conv/linear/softmax -> deployable. Thin copy of train_student_phaseB.py: same
# loader / run_model_training loop; only the model (and its front-end flags) differ.
#
# Validate on NORMAL data first (default --extra-per-class 0); the extra-data path
# (cap-first XC/TAU) is the same as Phase B and meant for the cluster run.
#
#   local validation (normal data):
#     .venv/bin/python experiments/modulation_arch/train_strf.py --epochs 60 --batch 16
#   frozen-prior ablation (Gabor STRF not learned):
#     .venv/bin/python experiments/modulation_arch/train_strf.py --epochs 60 --batch 16 --freeze-strf
#   cluster (full extra data):
#     .venv/bin/python experiments/modulation_arch/train_strf.py --epochs 60 --batch 16 --extra-per-class 3000

import sys, glob, argparse, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.model_tiny_ml import StrfBaseline
from pipeline_pytorch.model_training import run_model_training, run_validation_epoch
from pipeline_pytorch.paths import MODELS_DIR

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'
EXTRA_CACHE = ROOT / 'output' / '02_features' / 'cache_pcen_extra'
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'
XC_DEFAULT = '/home/hguimaraes/datasets/extra/xc'
BG_DEFAULT = '/home/hguimaraes/datasets/extra/background'


def build_index(sources):
  idx = {}
  for si, p in enumerate(sources):
    for r, s in enumerate(np.load(p)['stems']):
      idx[str(s)] = (si, r)
  return idx


def gather_rows(sources, idx, stems, key, dim):
  out = np.empty((len(stems), dim), np.float32)
  by_src = defaultdict(list)
  for di, s in enumerate(stems):
    si, r = idx[s]; by_src[si].append((di, r))
  for si, pairs in by_src.items():
    arr = np.load(sources[si])[key]
    dst = np.fromiter((d for d, _ in pairs), int, len(pairs))
    src = np.fromiter((r for _, r in pairs), int, len(pairs))
    out[dst] = np.asarray(arr[src], np.float32)
    del arr
  return out


def gather(split_glob, classes_idx, have, cap=0, seed=42, is_extra=False):
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


class LazyDS(torch.utils.data.Dataset):
  def __init__(self, paths, y, Tlog=None, Temb=None):
    self.paths, self.y, self.Tlog, self.Temb = paths, y, Tlog, Temb
  def __len__(self): return len(self.y)
  def __getitem__(self, i):
    x = torch.from_numpy(np.load(self.paths[i])['x'].reshape(1, 40, 133).astype(np.float32))
    yy = torch.tensor(int(self.y[i])); sid = torch.tensor(i)
    if self.Tlog is None:
      return x, yy, sid
    return x, yy, sid, torch.from_numpy(self.Tlog[i]), torch.from_numpy(self.Temb[i])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=16, help='D2 used 16; matters a lot per-epoch')
  ap.add_argument('--num-workers', type=int, default=4)
  # model (STRF front-end)
  ap.add_argument('--n-filters', type=int, default=32)
  ap.add_argument('--strf-kernel', type=int, default=9, help='Gabor STRF kernel size (odd, square)')
  ap.add_argument('--freeze-strf', action='store_true', help='freeze the Gabor STRF (pure prior, no learning)')
  ap.add_argument('--dropout', type=float, default=0.05)
  # data: 0 = normal/original-only (validate here); >0 = cap N/class extra; <0 = all
  ap.add_argument('--extra-per-class', type=int, default=0,
                  help='0 = normal data (original-only); N>0 = cap N/class extra; <0 = all')
  ap.add_argument('--xc', type=Path, default=XC_DEFAULT)
  ap.add_argument('--bg', type=Path, default=BG_DEFAULT)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)

  ck = torch.load(TEACHER / 'teacher_mlp' / 'teacher_head.pt', map_location='cpu', weights_only=False)
  classes = [str(n) for n in ck['label_names']]; cidx = {c: i for i, c in enumerate(classes)}; n_cls = len(classes)

  use_extra = args.extra_per_class != 0
  EMB_SOURCES = [TEACHER / 'Train.npz']
  LOGIT_SOURCES = [TEACHER / 'teacher_mlp' / 'soft_logits_Train.npz']
  if use_extra:
    EMB_SOURCES += [args.xc / 'embeddings' / 'perch_v2_cpu' / 'clips.npz', args.bg / 'embeddings' / 'perch_v2_cpu' / 'clips.npz']
    LOGIT_SOURCES += [args.xc / 'predictions.npz', args.bg / 'predictions.npz']
  emb_idx = build_index(EMB_SOURCES); logit_idx = build_index(LOGIT_SOURCES)
  have = lambda s: s in emb_idx and s in logit_idx

  orig = gather(str(CACHE / 'Train' / '*' / '*.npz'), cidx, have)
  if use_extra:
    cap = args.extra_per_class if args.extra_per_class > 0 else 0
    extra = gather(str(EXTRA_CACHE / '*' / '*.npz'), cidx, have, cap=cap, seed=args.seed, is_extra=True)
  else:
    extra = []
  items = orig + extra
  stems = [s for _, _, s in items]; paths = [str(f) for f, _, _ in items]
  y = np.array([ci for _, ci, _ in items], np.int64)
  Tlog = gather_rows(LOGIT_SOURCES, logit_idx, stems, 'logits', n_cls)
  Temb = gather_rows(EMB_SOURCES, emb_idx, stems, 'embeddings', 1536)
  Temb = (Temb - Temb.mean(0, keepdims=True)) / (Temb.std(0, keepdims=True) + 1e-6)

  vitems = [(str(f), cidx[f.parent.name]) for f in sorted((CACHE / 'Validation').glob('*/*.npz')) if f.parent.name in cidx]
  vpaths = [p for p, _ in vitems]; yv = np.array([c for _, c in vitems], np.int64)
  cap_label = 'none' if args.extra_per_class == 0 else ('all' if args.extra_per_class < 0 else args.extra_per_class)
  print('train: {} original + {} extra = {} (extra/class {}) | val: {} | classes {}'.format(
      len(orig), len(extra), len(items), cap_label, len(yv), n_cls))

  dl_tr = torch.utils.data.DataLoader(LazyDS(paths, y, Tlog, Temb), batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
  dl_va = torch.utils.data.DataLoader(LazyDS(vpaths, yv), batch_size=128, shuffle=False, num_workers=args.num_workers)

  model = StrfBaseline(input_shape=[1, 40, 133], num_classes=n_cls, save_path=str(MODELS_DIR),
                       n_filters=args.n_filters, strf_kernel=args.strf_kernel,
                       learnable_strf=not args.freeze_strf, dropout=args.dropout,
                       device={'use_cpu': not torch.cuda.is_available(), 'device_name': 'cuda:0'},
                       criterion={'module': 'torch.nn', 'attr': 'CrossEntropyLoss', 'kwargs': {'label_smoothing': 0.0}},
                       optimizer={'module': 'torch.optim', 'attr': 'Adam', 'kwargs': {'lr': 0.001, 'betas': [0.9, 0.999]}},
                       verbose=False)
  n_params = int(sum(p.numel() for p in model.parameters()))
  print('  StrfBaseline params: {:,} ({:.2f}x Baseline 97k) | strf_kernel {} learnable_strf {}'.format(
      n_params, n_params / 97000, args.strf_kernel, not args.freeze_strf))

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
  print('\n=== StrfBaseline (strf_kernel {} learnable {}) ==='.format(args.strf_kernel, not args.freeze_strf))
  print('  val acc {:.4f} auc {:.4f} | params {}  (D2 Baseline 0.6952 / 0.9402)'.format(acc, auc, n_params))

  out = Path(__file__).parent / 'results'; out.mkdir(parents=True, exist_ok=True)
  tag = 'strfbase_k{}_strf{}_x{}_s{}'.format(args.strf_kernel, int(not args.freeze_strf), args.extra_per_class, args.seed)
  import yaml
  yaml.safe_dump({
    'model': 'StrfBaseline', 'n_filters': args.n_filters, 'strf_kernel': args.strf_kernel,
    'learnable_strf': not args.freeze_strf, 'dropout': args.dropout,
    'extra_per_class': args.extra_per_class, 'batch': args.batch, 'epochs': args.epochs, 'seed': args.seed,
    'params': n_params, 'val_acc': round(float(acc), 4), 'val_auc': round(float(auc), 4),
    'finished_utc': datetime.datetime.utcnow().isoformat(),
  }, open(out / (tag + '.yaml'), 'w'), sort_keys=False)
  print('  results -> {}/{}.yaml'.format(out, tag))


if __name__ == '__main__':
  main()
