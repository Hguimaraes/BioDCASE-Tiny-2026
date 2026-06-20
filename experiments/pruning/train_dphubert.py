# --
# DPHuBERT-style joint distillation + structured pruning of a gated EfficientNet
# down to a target parameter budget, distilling from Perch 2.0.
#
# 3 stages (run all with --stage all):
#   1. distill + prune: GatedEffNet with expand-channel L0 gates. Loss =
#      distillation (logit KD from Perch + optional single spatial hint) + a
#      size-targeting Lagrangian that drives expected params -> --target-params.
#      Optimizer groups: weights (lr), gates log_alpha (+reg_lr), Lagrange
#      multipliers lambda1/2 (-reg_lr, i.e. gradient ascent).
#   2. prune: model.to_pruned() -> compact, gate-free model (~target size).
#   3. final distill: retrain the pruned model (no gates / no Lagrangian).
#
# Student input: 40-mel PCEN (cache_pcen). Teacher: Perch v2 logits + the global
# 1536-d embedding regressed from the student's GAP descriptor (--embed-distill,
# the proven D2 lever, on by default). Optional spatial_embedding hint via
# --feature-distill on (Track G, interpolated to the student grid).
#
# Extra teacher-labeled data (Phase B): pass --extra-per-class N to add
# agreement-filtered Xeno-canto + TAU background clips (cache_pcen_extra features
# + XC/BG predictions.npz logits) to the distillation set, capped at N/class.
# Incompatible with --feature-distill on (no spatial hints exported for extra).
#
#   .venv/bin/python experiments/pruning/train_dphubert.py --stage all \
#       --target-params 120000 --base-width 0.35 --feature-distill off \
#       --extra-per-class 3000
#
# Results (summary YAML + per-epoch CSV log) are written under
# experiments/pruning/results/, not just printed.

import sys, time, argparse, csv, yaml, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.distillation import distillation_loss, embedding_hint_loss
from experiments.pruning.gated_effnet import GatedEffNet

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'              # 40-mel PCEN
EXTRA_CACHE = ROOT / 'output' / '02_features' / 'cache_pcen_extra'  # Phase B extra
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'
XC_DEFAULT = '/home/hguimaraes/datasets/extra/xc'
BG_DEFAULT = '/home/hguimaraes/datasets/extra/background'


def load_split(split):
  files = sorted((CACHE / split).glob('*/*.npz'))
  xs, ys, stems = [], [], []
  for f in files:
    d = np.load(f)
    xs.append(d['x'].astype(np.float32).reshape(1, 40, 133))
    ys.append(f.parent.name); stems.append(f.stem)
  return np.stack(xs), np.array(ys), stems


def build_index(sources):
  """stem -> (source_idx, row). Reads ONLY the small `stems` array from each npz,
  so membership/capping happens before the big arrays are decompressed."""
  idx = {}
  for si, p in enumerate(sources):
    for r, s in enumerate(np.load(p)['stems']):
      idx[str(s)] = (si, r)
  return idx


def gather_rows(sources, idx, stems, key, dim):
  """(len(stems), dim) teacher tensors for `key`; each source is decompressed once
  and only the rows we need are kept."""
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


def gather_extra(cache, have, cap, seed):
  """(path, class_name, stem) for extra clips with teacher logits, capped per class."""
  rng = np.random.default_rng(seed)
  by_cls = defaultdict(list)
  for f in sorted(cache.glob('*/*.npz')):
    if have(f.stem): by_cls[f.parent.name].append(f)
  items = []
  for cls, fs in by_cls.items():
    if cap and len(fs) > cap:
      fs = [fs[i] for i in rng.choice(len(fs), cap, replace=False)]
    items += [(f, cls, f.stem) for f in fs]
  return items


def stack_features(items):
  return np.stack([np.load(f)['x'].astype(np.float32).reshape(1, 40, 133) for f, _, _ in items])


def align_teacher(t, hw):
  Hs, Ws = hw
  if t.shape[-2:] == (Ws, Hs) and Hs != Ws:
    t = t.transpose(-1, -2)
  if t.shape[-2:] != (Hs, Ws):
    t = F.interpolate(t, size=(Hs, Ws), mode='bilinear', align_corners=False)
  return t


def feature_hint_loss(s_feat, t_feat, proj):
  s = proj(s_feat); t = align_teacher(t_feat, s.shape[-2:])
  return F.l1_loss(s, t) - F.logsigmoid(F.cosine_similarity(s, t, dim=1)).mean()


def macro_auc(logits, y, n):
  from sklearn.metrics import roc_auc_score
  return roc_auc_score(np.eye(n)[y], torch.softmax(torch.from_numpy(logits), 1).numpy(),
                       average='macro', multi_class='ovr')


def evaluate(model, Xva, yva, n, dev, batch=64):
  model.eval(); vl = []
  with torch.no_grad():
    for i in range(0, len(Xva), batch):
      vl.append(model(Xva[i:i + batch].to(dev)).cpu())
  vl = torch.cat(vl).numpy()
  return float((vl.argmax(1) == yva).mean()), macro_auc(vl, yva, n)


def distill_step(model, xb, yb, Tlog_b, n, feat_proj, Tsp_b, embed_proj=None, Temb_b=None, embed_weight=1.0):
  need_feat = feat_proj is not None or embed_proj is not None
  logits, s_feat = model.forward_with_spatial(xb) if need_feat else (model(xb), None)
  hard = F.one_hot(yb, n).float()
  loss, _ = distillation_loss(logits, Tlog_b, hard, alpha=0.5, temperature=3.0, label_smoothing=0.1)
  if feat_proj is not None:                                       # spatial-map hint (Track G)
    loss = loss + feature_hint_loss(s_feat, Tsp_b, feat_proj)
  if embed_proj is not None:                                      # global 1536-d embed regress (D2 lever)
    loss = loss + embed_weight * embedding_hint_loss(embed_proj(s_feat.mean((2, 3))), Temb_b)
  return loss


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--stage', choices=['1', '2', 'all'], default='all')
  ap.add_argument('--target-params', type=int, default=120000)
  ap.add_argument('--base-width', type=float, default=0.35)
  ap.add_argument('--feature-distill', choices=['on', 'off'], default='off', help='Perch spatial-map hint (Track G)')
  ap.add_argument('--embed-distill', choices=['on', 'off'], default='on', help='regress Perch global 1536-d embedding (the D2 lever)')
  ap.add_argument('--embed-weight', type=float, default=1.0)
  ap.add_argument('--extra-per-class', type=int, default=0, help='add Phase B extra clips, capped per class (0 = off)')
  ap.add_argument('--xc', type=Path, default=XC_DEFAULT, help='Xeno-canto extra-data root (predictions.npz)')
  ap.add_argument('--bg', type=Path, default=BG_DEFAULT, help='background extra-data root (predictions.npz)')
  ap.add_argument('--epochs', type=int, default=100)
  ap.add_argument('--final-epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=64)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--reg-lr', type=float, default=0.02)
  ap.add_argument('--sparsity-warmup-frac', type=float, default=0.3)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  if args.feature_distill == 'on' and args.extra_per_class:
    ap.error('--feature-distill on is unsupported with --extra-per-class (no spatial hints for extra clips)')
  torch.manual_seed(args.seed); np.random.seed(args.seed)
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'

  # ---- assemble training set: original (all) + optional capped extra ----
  embed_on = args.embed_distill == 'on'
  logit_sources = [TEACHER / 'teacher_mlp' / 'soft_logits_Train.npz']
  embed_sources = [TEACHER / 'Train.npz']
  if args.extra_per_class:
    logit_sources += [args.xc / 'predictions.npz', args.bg / 'predictions.npz']
    embed_sources += [args.xc / 'embeddings' / 'perch_v2_cpu' / 'clips.npz',
                      args.bg / 'embeddings' / 'perch_v2_cpu' / 'clips.npz']
  lidx = build_index(logit_sources)
  eidx = build_index(embed_sources) if embed_on else None
  have = lambda s: s in lidx and (eidx is None or s in eidx)

  orig = [(f, f.parent.name, f.stem) for f in sorted((CACHE / 'Train').glob('*/*.npz')) if have(f.stem)]
  extra = gather_extra(EXTRA_CACHE, have, args.extra_per_class, args.seed) if args.extra_per_class else []
  items = orig + extra

  Xva, yva_n, _ = load_split('Validation')
  ytr_n = np.array([c for _, c, _ in items]); stems = [s for _, _, s in items]
  classes = sorted(set(ytr_n) | set(yva_n)); c2i = {c: i for i, c in enumerate(classes)}; n = len(classes)
  ytr = np.array([c2i[c] for c in ytr_n]); yva = np.array([c2i[c] for c in yva_n])
  Xtr = stack_features(items)
  Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr).long(); Xva_t = torch.from_numpy(Xva)
  Tlog = torch.from_numpy(gather_rows(logit_sources, lidx, stems, 'logits', n))
  Temb = None
  if embed_on:
    Temb = gather_rows(embed_sources, eidx, stems, 'embeddings', 1536)
    Temb = (Temb - Temb.mean(0, keepdims=True)) / (Temb.std(0, keepdims=True) + 1e-6)   # z-score (as D2)
    Temb = torch.from_numpy(Temb)
  print('Train {} ({} original + {} extra, cap {}/class) Val {} | classes {} | device {} | feature_distill {} | embed_distill {}'.format(
      Xtr.shape, len(orig), len(extra), args.extra_per_class or 'off', Xva.shape, n, dev, args.feature_distill, args.embed_distill))

  # teacher spatial hint (optional; original clips only)
  Tsp = feat_proj = None
  if args.feature_distill == 'on':
    d = np.load(TEACHER / 'spatial' / 'Train.npz', allow_pickle=True)
    s2s = {str(s): d['spatial'][i] for i, s in enumerate(d['stems'])}
    Tsp = np.ascontiguousarray(np.stack([s2s[s] for s in stems]).transpose(0, 3, 1, 2)).astype(np.float32)
    print('  teacher spatial (NCHW):', Tsp.shape)

  stage1_log, final_log = [], []

  # ---------- Stage 1: distill + prune ----------
  model = GatedEffNet(num_classes=n, width=args.base_width, use_gate=True).to(dev)
  original = float(model.get_num_params().item())
  target_sparsity = 1.0 - args.target_params / original
  print('Stage 1 | base params {:.0f} | target {} -> sparsity {:.3f}'.format(
      original, args.target_params, target_sparsity))
  if args.feature_distill == 'on':
    feat_proj = nn.Conv2d(model.head_ch, Tsp.shape[1], 1).to(dev)
  # global-embedding projection: GAP descriptor (head_ch) -> Perch 1536-d. head_ch is
  # stable through pruning, so this same head works in stage 1 and the final distill.
  embed_proj = nn.Linear(model.head_ch, 1536).to(dev) if embed_on else None

  lam1 = nn.Parameter(torch.zeros((), device=dev)); lam2 = nn.Parameter(torch.zeros((), device=dev))
  main_p = [p for nm, p in model.named_parameters() if 'log_alpha' not in nm]
  if feat_proj is not None: main_p += list(feat_proj.parameters())
  if embed_proj is not None: main_p += list(embed_proj.parameters())
  gate_p = [p for nm, p in model.named_parameters() if 'log_alpha' in nm]
  opt = torch.optim.AdamW([
    {'params': main_p, 'lr': args.lr, 'weight_decay': 1e-4},
    {'params': gate_p, 'lr': args.reg_lr, 'weight_decay': 0.0},
    {'params': [lam1, lam2], 'lr': -args.reg_lr, 'weight_decay': 0.0},   # ascent on multipliers
  ])
  total_steps = args.epochs * ((len(Xtr_t) + args.batch - 1) // args.batch)
  warmup_steps = max(1, int(args.sparsity_warmup_frac * total_steps))
  step = 0
  for epoch in range(args.epochs):
    model.train(); perm = torch.randperm(len(Xtr_t)); t0 = time.time()
    for i in range(0, len(perm), args.batch):
      idx = perm[i:i + args.batch]
      xb = Xtr_t[idx].to(dev); yb = ytr_t[idx].to(dev)
      Tsp_b = torch.from_numpy(Tsp[idx.numpy()]).to(dev) if Tsp is not None else None
      Temb_b = Temb[idx].to(dev) if Temb is not None else None
      opt.zero_grad()
      loss = distill_step(model, xb, yb, Tlog[idx].to(dev), n, feat_proj, Tsp_b, embed_proj, Temb_b, args.embed_weight)
      tgt = target_sparsity * min(1.0, step / warmup_steps)
      exp_sp = 1.0 - model.get_num_params() / original
      loss_reg = lam1 * (exp_sp - tgt) + lam2 * (exp_sp - tgt) ** 2
      (loss + loss_reg).backward(); opt.step(); step += 1
    exp_sp_now = float(1.0 - model.get_num_params().item() / original)
    stage1_log.append({'epoch': epoch, 'distill_loss': round(float(loss), 4),
                       'exp_sparsity': round(exp_sp_now, 4), 'target_sparsity': round(tgt, 4),
                       'approx_params': int(original * (1 - exp_sp_now))})
    if epoch % 5 == 0 or epoch == args.epochs - 1:
      print('  e{:3d} distill {:.3f} | exp_sparsity {:.3f} (tgt {:.3f}) | ~params {:.0f} | {:.0f}s'.format(
          epoch, float(loss), exp_sp_now, tgt, original * (1 - exp_sp_now), time.time() - t0))

  # ---------- Stage 2: prune + final distill ----------
  model.eval()
  pruned = model.to_pruned().to(dev)
  pruned_params = sum(p.numel() for p in pruned.parameters())
  print('Pruned model: {} params (target {})'.format(pruned_params, args.target_params))
  acc0, auc0 = evaluate(pruned, Xva_t, yva, n, dev)
  print('  right after prune (no finetune): acc {:.4f} auc {:.4f}'.format(acc0, auc0))

  opt2_params = list(pruned.parameters()) + (list(embed_proj.parameters()) if embed_proj is not None else [])
  opt2 = torch.optim.Adam(opt2_params, lr=args.lr, weight_decay=1e-4)
  sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.final_epochs)
  best = {'acc': 0.0, 'auc': float('nan'), 'epoch': 0}        # best by acc (kept for continuity)
  best_auc = {'acc': 0.0, 'auc': 0.0, 'epoch': 0}            # best by auc (our checkpoint metric)
  for epoch in range(args.final_epochs):
    pruned.train(); perm = torch.randperm(len(Xtr_t)); tl = []
    for i in range(0, len(perm), args.batch):
      idx = perm[i:i + args.batch]
      xb = Xtr_t[idx].to(dev); yb = ytr_t[idx].to(dev)
      Temb_b = Temb[idx].to(dev) if Temb is not None else None
      opt2.zero_grad()
      loss = distill_step(pruned, xb, yb, Tlog[idx].to(dev), n, None, None, embed_proj, Temb_b, args.embed_weight)
      loss.backward(); opt2.step(); tl.append(float(loss))
    sched2.step()
    acc, auc = evaluate(pruned, Xva_t, yva, n, dev)
    final_log.append({'epoch': epoch + 1, 'train_loss': round(float(np.mean(tl)), 4),
                      'val_acc': round(acc, 4), 'val_auc': round(auc, 4)})
    if acc > best['acc']: best = {'acc': acc, 'auc': auc, 'epoch': epoch + 1}
    if auc > best_auc['auc']: best_auc = {'acc': acc, 'auc': auc, 'epoch': epoch + 1}
  print('\n=== DPHuBERT-pruned EffNet ({} params) ==='.format(pruned_params))
  print('  best-acc val acc {:.4f} auc {:.4f} (epoch {})'.format(best['acc'], best['auc'], best['epoch']))
  print('  best-auc val acc {:.4f} auc {:.4f} (epoch {})'.format(best_auc['acc'], best_auc['auc'], best_auc['epoch']))
  print('  reference: PCEN-Baseline 0.6497 / 0.9296 (97k params)')

  # ---------- write results to disk (summary YAML + per-epoch CSV log) ----------
  out = Path(__file__).parent / 'results'; out.mkdir(parents=True, exist_ok=True)
  tag = 'dphubert_{}p_{}_e{}_x{}_s{}'.format(args.target_params, args.feature_distill, args.embed_distill, args.extra_per_class, args.seed)
  yaml.safe_dump({
    'model': 'dphubert_gated_effnet', 'target_params': args.target_params,
    'pruned_params': int(pruned_params), 'base_width': args.base_width,
    'feature_distill': args.feature_distill, 'embed_distill': args.embed_distill,
    'embed_weight': args.embed_weight, 'seed': args.seed,
    'extra_per_class': args.extra_per_class, 'n_train': len(items),
    'n_original': len(orig), 'n_extra': len(extra),
    'epochs': args.epochs, 'final_epochs': args.final_epochs, 'lr': args.lr,
    'val_acc': round(best['acc'], 4), 'val_auc': round(best['auc'], 4), 'best_epoch': best['epoch'],
    'val_acc_at_best_auc': round(best_auc['acc'], 4), 'val_auc_best': round(best_auc['auc'], 4),
    'best_auc_epoch': best_auc['epoch'],
    'acc_after_prune_nofinetune': round(acc0, 4), 'auc_after_prune_nofinetune': round(auc0, 4),
    'finished_utc': datetime.datetime.utcnow().isoformat(),
  }, open(out / (tag + '.yaml'), 'w'), sort_keys=False)
  with open(out / (tag + '_finaldistill_log.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['epoch', 'train_loss', 'val_acc', 'val_auc']); w.writeheader(); w.writerows(final_log)
  with open(out / (tag + '_stage1_log.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['epoch', 'distill_loss', 'exp_sparsity', 'target_sparsity', 'approx_params'])
    w.writeheader(); w.writerows(stage1_log)
  print('  results -> {}/{}.yaml (+ _finaldistill_log.csv, _stage1_log.csv)'.format(out, tag))


if __name__ == '__main__':
  main()
