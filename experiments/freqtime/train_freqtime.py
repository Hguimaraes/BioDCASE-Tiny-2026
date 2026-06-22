# --
# FreqTimeNet (step 1): frequency/time-SEPARATED tiny CNN (Tan et al. 2019, adapted to
# classification). FREQUENCY module (GaborSTRF front + freq-dilated 3x3 convs) collapses
# the mel axis while KEEPING the 133-frame time axis; TIME module of SIMPLE dilated 1-D
# conv residual blocks (no branching/gating/FiLM/Fourier -- those are deferred) models
# rhythm; mean+std temporal pooling -> classifier. Trained on the proven D2 recipe over
# cache_pcen (PCEN + Perch logit KD + 1536-d embedding distillation + EMA).
#
# Recipe is the CANONICAL D2 (batch 16, 120 epochs, min_lr_factor 0.05) -- baked in so we
# don't repeat the 60-epoch undertraining bug. Compare to StrfBaseline @120ep (0.7098/0.9485).
#
#   .venv/bin/python experiments/freqtime/train_freqtime.py --seed 1
#   (repeat --seed 2, 3)

import sys, glob, argparse, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.model_tiny_ml import FreqTimeNet
from pipeline_pytorch.model_training import run_model_training, run_validation_epoch
from pipeline_pytorch.paths import MODELS_DIR

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'


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


def gather(split_glob, classes_idx, have):
  items = []
  for f in sorted(Path(p) for p in glob.glob(split_glob)):
    cls = f.parent.name
    if cls not in classes_idx or not have(f.stem):
      continue
    items.append((f, classes_idx[cls], f.stem))
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
  ap.add_argument('--epochs', type=int, default=120, help='canonical D2 (best_epoch ~119)')
  ap.add_argument('--batch', type=int, default=16, help='canonical D2 batch')
  ap.add_argument('--num-workers', type=int, default=4)
  ap.add_argument('--n-filters', type=int, default=32, help='STRF front channels')
  ap.add_argument('--tcn-ch', type=int, default=64, help='freq-module body + time-module channels')
  ap.add_argument('--tcn-kernel', type=int, default=7)
  ap.add_argument('--tcn-dilations', type=int, nargs='+', default=[1, 2, 4, 8])
  ap.add_argument('--no-strf', action='store_true', help='plain 9x9 conv front instead of Gabor STRF')
  ap.add_argument('--dropout', type=float, default=0.05)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)

  ck = torch.load(TEACHER / 'teacher_mlp' / 'teacher_head.pt', map_location='cpu', weights_only=False)
  classes = [str(n) for n in ck['label_names']]; cidx = {c: i for i, c in enumerate(classes)}; n_cls = len(classes)

  EMB_SOURCES = [TEACHER / 'Train.npz']
  LOGIT_SOURCES = [TEACHER / 'teacher_mlp' / 'soft_logits_Train.npz']
  emb_idx = build_index(EMB_SOURCES); logit_idx = build_index(LOGIT_SOURCES)
  have = lambda s: s in emb_idx and s in logit_idx

  items = gather(str(CACHE / 'Train' / '*' / '*.npz'), cidx, have)
  stems = [s for _, _, s in items]; paths = [str(f) for f, _, _ in items]
  y = np.array([ci for _, ci, _ in items], np.int64)
  Tlog = gather_rows(LOGIT_SOURCES, logit_idx, stems, 'logits', n_cls)
  Temb = gather_rows(EMB_SOURCES, emb_idx, stems, 'embeddings', 1536)
  Temb = (Temb - Temb.mean(0, keepdims=True)) / (Temb.std(0, keepdims=True) + 1e-6)

  vitems = [(str(f), cidx[f.parent.name]) for f in sorted((CACHE / 'Validation').glob('*/*.npz')) if f.parent.name in cidx]
  vpaths = [p for p, _ in vitems]; yv = np.array([c for _, c in vitems], np.int64)
  print('train: {} | val: {} | classes {}'.format(len(items), len(yv), n_cls))

  dl_tr = torch.utils.data.DataLoader(LazyDS(paths, y, Tlog, Temb), batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
  dl_va = torch.utils.data.DataLoader(LazyDS(vpaths, yv), batch_size=128, shuffle=False, num_workers=args.num_workers)

  model = FreqTimeNet(input_shape=[1, 40, 133], num_classes=n_cls, save_path=str(MODELS_DIR),
                      n_filters=args.n_filters, tcn_ch=args.tcn_ch, tcn_kernel=args.tcn_kernel,
                      tcn_dilations=args.tcn_dilations, strf=not args.no_strf, dropout=args.dropout,
                      device={'use_cpu': not torch.cuda.is_available(), 'device_name': 'cuda:0'},
                      criterion={'module': 'torch.nn', 'attr': 'CrossEntropyLoss', 'kwargs': {'label_smoothing': 0.0}},
                      optimizer={'module': 'torch.optim', 'attr': 'Adam', 'kwargs': {'lr': 0.001, 'betas': [0.9, 0.999]}},
                      verbose=False)
  n_params = int(sum(p.numel() for p in model.parameters()))
  rf = 1 + sum((args.tcn_kernel - 1) * d for d in args.tcn_dilations)
  print('  FreqTimeNet params: {:,} ({:.2f}x Baseline 97k) | tcn_ch {} kernel {} dilations {} (RF {} frames) strf {}'.format(
      n_params, n_params / 97000, args.tcn_ch, args.tcn_kernel, args.tcn_dilations, rf, not args.no_strf))

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
  print('\n=== FreqTimeNet (freq/time-separated, simple 1-D trunk) ===')
  print('  val acc {:.4f} auc {:.4f} | params {}  (StrfBaseline @120ep 0.7098 / 0.9485)'.format(acc, auc, n_params))

  out = Path(__file__).parent / 'results'; out.mkdir(parents=True, exist_ok=True)
  tag = 'freqtime_s{}'.format(args.seed)
  import yaml
  yaml.safe_dump({
    'model': 'FreqTimeNet', 'n_filters': args.n_filters, 'tcn_ch': args.tcn_ch, 'tcn_kernel': args.tcn_kernel,
    'tcn_dilations': args.tcn_dilations, 'strf': not args.no_strf, 'dropout': args.dropout,
    'batch': args.batch, 'epochs': args.epochs, 'seed': args.seed, 'params': n_params,
    'val_acc': round(float(acc), 4), 'val_auc': round(float(auc), 4),
    'finished_utc': datetime.datetime.utcnow().isoformat(),
  }, open(out / (tag + '.yaml'), 'w'), sort_keys=False)
  print('  results -> {}/{}.yaml'.format(out, tag))


if __name__ == '__main__':
  main()
