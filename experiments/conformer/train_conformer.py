# --
# Architecture probe: the PROVEN D2 recipe (PCEN student + Perch logit KD +
# 1536-d embedding distillation + EMA) with a small Conformer backbone instead of
# the Baseline CNN. To avoid reimplementing the winning recipe, this is a thin
# copy of train_student_phaseB.py: same teacher-labeled dataset, same
# run_model_training loop. The ONLY change is the model -> ConformerStudent
# (torchaudio Conformer; 40 mel bins -> d_model, 133 frames as the sequence,
# time-pooled descriptor carries the embedding-distillation lever).
#
# Recipe is identical to D2 (0.6952 / 0.9402), so this is a clean
# architecture-only comparison. tflite export is deferred (attention/conv graph).
#
#   .venv/bin/python experiments/conformer/train_conformer.py --epochs 60
#   .venv/bin/python experiments/conformer/train_conformer.py --epochs 60 --d-model 72 --num-layers 4

import sys, argparse, datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.model_tiny_ml import ConformerStudent
from pipeline_pytorch.model_training import run_model_training, run_validation_epoch
from pipeline_pytorch.distillation import load_teacher_logits, load_teacher_embeddings
from pipeline_pytorch.paths import MODELS_DIR

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'


class TupleDS(torch.utils.data.Dataset):
  """yields (x, y, sid[, t_logits, t_emb]); with teacher -> the D2 embed path."""
  def __init__(self, X, y, Tlog=None, Temb=None):
    self.X, self.y, self.Tlog, self.Temb = X, y, Tlog, Temb
  def __len__(self): return len(self.y)
  def __getitem__(self, i):
    x = torch.from_numpy(self.X[i]); yy = torch.tensor(int(self.y[i])); sid = torch.tensor(i)
    if self.Tlog is None:
      return x, yy, sid
    return x, yy, sid, torch.from_numpy(self.Tlog[i]), torch.from_numpy(self.Temb[i])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=64)
  ap.add_argument('--d-model', type=int, default=64)
  ap.add_argument('--num-layers', type=int, default=4)
  ap.add_argument('--num-heads', type=int, default=4)
  ap.add_argument('--ffn-dim', type=int, default=128)
  ap.add_argument('--conv-kernel', type=int, default=15, help='depthwise conv kernel (odd)')
  ap.add_argument('--dropout', type=float, default=0.1)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  assert args.d_model % args.num_heads == 0, 'd_model must be divisible by num_heads'
  torch.manual_seed(args.seed); np.random.seed(args.seed)

  # canonical 11-class order = teacher head label_names (aligns logit columns)
  ck = torch.load(TEACHER / 'teacher_mlp' / 'teacher_head.pt', map_location='cpu', weights_only=False)
  classes = [str(c) for c in ck['label_names']]; cidx = {c: i for i, c in enumerate(classes)}; n_cls = len(classes)

  logit_d = load_teacher_logits(str(TEACHER / 'teacher_mlp'), 'Train')
  emb_d, _ = load_teacher_embeddings(str(TEACHER), 'Train', standardize=True)

  items = [(f, cidx[f.parent.name]) for f in sorted((CACHE / 'Train').glob('*/*.npz'))
           if f.parent.name in cidx and f.stem in logit_d and f.stem in emb_d]
  N = len(items)
  X = np.empty((N, 1, 40, 133), np.float32); y = np.empty(N, np.int64)
  Tlog = np.empty((N, n_cls), np.float32); Temb = np.empty((N, 1536), np.float32)
  for i, (f, ci) in enumerate(items):
    X[i] = np.load(f)['x'].reshape(1, 40, 133); y[i] = ci
    Tlog[i] = logit_d[f.stem]; Temb[i] = emb_d[f.stem]

  vitems = [(f, cidx[f.parent.name]) for f in sorted((CACHE / 'Validation').glob('*/*.npz')) if f.parent.name in cidx]
  Xv = np.stack([np.load(f)['x'].reshape(1, 40, 133) for f, _ in vitems]).astype(np.float32)
  yv = np.array([c for _, c in vitems], np.int64)
  print('train: {} | val: {} | classes {} | conformer d{} L{} h{} ffn{} k{}'.format(
      N, len(yv), n_cls, args.d_model, args.num_layers, args.num_heads, args.ffn_dim, args.conv_kernel))

  dl_tr = torch.utils.data.DataLoader(TupleDS(X, y, Tlog, Temb), batch_size=args.batch, shuffle=True)
  dl_va = torch.utils.data.DataLoader(TupleDS(Xv, yv), batch_size=128, shuffle=False)

  # same construction as train_student_phaseB.py; only the model class + its kwargs differ
  model = ConformerStudent(input_shape=[1, 40, 133], num_classes=n_cls, save_path=str(MODELS_DIR),
                           d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads,
                           ffn_dim=args.ffn_dim, conv_kernel=args.conv_kernel, dropout=args.dropout,
                           device={'use_cpu': not torch.cuda.is_available(), 'device_name': 'cuda:0'},
                           criterion={'module': 'torch.nn', 'attr': 'CrossEntropyLoss', 'kwargs': {'label_smoothing': 0.0}},
                           optimizer={'module': 'torch.optim', 'attr': 'Adam', 'kwargs': {'lr': 0.001, 'betas': [0.9, 0.999]}},
                           verbose=False)
  n_params = int(sum(p.numel() for p in model.parameters()))
  print('  model params: {:,} ({:.1f}x Baseline 97k)'.format(n_params, n_params / 97000))

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
  print('\n=== Conformer student (d{} L{}) ==='.format(args.d_model, args.num_layers))
  print('  val acc {:.4f} auc {:.4f} | params {}  (D2 Baseline 0.6952 / 0.9402)'.format(acc, auc, n_params))

  out = Path(__file__).parent / 'results'; out.mkdir(parents=True, exist_ok=True)
  tag = 'conformer_d{}_L{}_s{}'.format(args.d_model, args.num_layers, args.seed)
  import yaml
  yaml.safe_dump({
    'model': 'ConformerStudent', 'd_model': args.d_model, 'num_layers': args.num_layers,
    'num_heads': args.num_heads, 'ffn_dim': args.ffn_dim, 'conv_kernel': args.conv_kernel,
    'dropout': args.dropout, 'seed': args.seed, 'params': n_params, 'epochs': args.epochs,
    'val_acc': round(float(acc), 4), 'val_auc': round(float(auc), 4),
    'finished_utc': datetime.datetime.utcnow().isoformat(),
  }, open(out / (tag + '.yaml'), 'w'), sort_keys=False)
  print('  results -> {}/{}.yaml'.format(out, tag))


if __name__ == '__main__':
  main()
