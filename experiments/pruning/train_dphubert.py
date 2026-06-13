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
# Student input: 40-mel PCEN (cache_pcen). Teacher: Perch v2 logits (+ optional
# spatial_embedding hint, interpolated to the student's small grid).
#
#   .venv/bin/python experiments/pruning/train_dphubert.py --stage all \
#       --target-params 120000 --base-width 0.35 --feature-distill off

import sys, time, argparse, numpy as np
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_pytorch.distillation import distillation_loss, load_teacher_logits
from experiments.pruning.gated_effnet import GatedEffNet

CACHE = ROOT / 'output' / '02_features' / 'cache_pcen'          # 40-mel PCEN
TEACHER = ROOT / 'experiments' / 'perch' / 'embeddings' / 'perch_v2_cpu'


def load_split(split):
  files = sorted((CACHE / split).glob('*/*.npz'))
  xs, ys, stems = [], [], []
  for f in files:
    d = np.load(f)
    xs.append(d['x'].astype(np.float32).reshape(1, 40, 133))
    ys.append(f.parent.name); stems.append(f.stem)
  return np.stack(xs), np.array(ys), stems


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


def distill_step(model, xb, yb, Tlog_b, n, feat_proj, Tsp_b):
  out = model.forward_with_spatial(xb) if feat_proj is not None else (model(xb), None)
  logits, s_feat = out
  hard = F.one_hot(yb, n).float()
  loss, _ = distillation_loss(logits, Tlog_b, hard, alpha=0.5, temperature=3.0, label_smoothing=0.1)
  if feat_proj is not None:
    loss = loss + feature_hint_loss(s_feat, Tsp_b, feat_proj)
  return loss


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--stage', choices=['1', '2', 'all'], default='all')
  ap.add_argument('--target-params', type=int, default=120000)
  ap.add_argument('--base-width', type=float, default=0.35)
  ap.add_argument('--feature-distill', choices=['on', 'off'], default='off')
  ap.add_argument('--epochs', type=int, default=100)
  ap.add_argument('--final-epochs', type=int, default=60)
  ap.add_argument('--batch', type=int, default=64)
  ap.add_argument('--lr', type=float, default=1e-3)
  ap.add_argument('--reg-lr', type=float, default=0.02)
  ap.add_argument('--sparsity-warmup-frac', type=float, default=0.3)
  ap.add_argument('--seed', type=int, default=1)
  args = ap.parse_args()
  torch.manual_seed(args.seed); np.random.seed(args.seed)
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'

  Xtr, ytr_n, stems = load_split('Train')
  Xva, yva_n, _ = load_split('Validation')
  classes = sorted(set(ytr_n) | set(yva_n)); c2i = {c: i for i, c in enumerate(classes)}
  ytr = np.array([c2i[c] for c in ytr_n]); yva = np.array([c2i[c] for c in yva_n]); n = len(classes)
  Xtr_t = torch.from_numpy(Xtr); ytr_t = torch.from_numpy(ytr).long(); Xva_t = torch.from_numpy(Xva)
  s2l = load_teacher_logits(str(TEACHER / 'teacher_mlp'), 'Train')
  Tlog = torch.tensor(np.stack([s2l[s] for s in stems]), dtype=torch.float32)
  print('Train {} Val {} | classes {} | device {} | feature_distill {}'.format(
      Xtr.shape, Xva.shape, n, dev, args.feature_distill))

  # teacher spatial hint (optional)
  Tsp = feat_proj = None
  if args.feature_distill == 'on':
    d = np.load(TEACHER / 'spatial' / 'Train.npz', allow_pickle=True)
    s2s = {str(s): d['spatial'][i] for i, s in enumerate(d['stems'])}
    Tsp = np.ascontiguousarray(np.stack([s2s[s] for s in stems]).transpose(0, 3, 1, 2)).astype(np.float32)
    print('  teacher spatial (NCHW):', Tsp.shape)

  # ---------- Stage 1: distill + prune ----------
  model = GatedEffNet(num_classes=n, width=args.base_width, use_gate=True).to(dev)
  original = float(model.get_num_params().item())
  target_sparsity = 1.0 - args.target_params / original
  print('Stage 1 | base params {:.0f} | target {} -> sparsity {:.3f}'.format(
      original, args.target_params, target_sparsity))
  if args.feature_distill == 'on':
    feat_proj = nn.Conv2d(model.head_ch, Tsp.shape[1], 1).to(dev)

  lam1 = nn.Parameter(torch.zeros((), device=dev)); lam2 = nn.Parameter(torch.zeros((), device=dev))
  main_p = [p for nm, p in model.named_parameters() if 'log_alpha' not in nm]
  if feat_proj is not None: main_p += list(feat_proj.parameters())
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
      opt.zero_grad()
      loss = distill_step(model, xb, yb, Tlog[idx].to(dev), n, feat_proj, Tsp_b)
      tgt = target_sparsity * min(1.0, step / warmup_steps)
      exp_sp = 1.0 - model.get_num_params() / original
      loss_reg = lam1 * (exp_sp - tgt) + lam2 * (exp_sp - tgt) ** 2
      (loss + loss_reg).backward(); opt.step(); step += 1
    exp_sp_now = float(1.0 - model.get_num_params().item() / original)
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

  opt2 = torch.optim.Adam(pruned.parameters(), lr=args.lr, weight_decay=1e-4)
  sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.final_epochs)
  best = {'acc': 0.0, 'auc': float('nan'), 'epoch': 0}
  for epoch in range(args.final_epochs):
    pruned.train(); perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(perm), args.batch):
      idx = perm[i:i + args.batch]
      xb = Xtr_t[idx].to(dev); yb = ytr_t[idx].to(dev)
      opt2.zero_grad()
      hard = F.one_hot(yb, n).float()
      loss, _ = distillation_loss(pruned(xb), Tlog[idx].to(dev), hard, alpha=0.5, temperature=3.0, label_smoothing=0.1)
      loss.backward(); opt2.step()
    sched2.step()
    acc, auc = evaluate(pruned, Xva_t, yva, n, dev)
    if acc > best['acc']: best = {'acc': acc, 'auc': auc, 'epoch': epoch + 1}
  print('\n=== DPHuBERT-pruned EffNet ({} params) ==='.format(pruned_params))
  print('  best val acc {:.4f} auc {:.4f} (epoch {})'.format(best['acc'], best['auc'], best['epoch']))
  print('  reference: PCEN-Baseline 0.6497 / 0.9296 (97k params)')

  import yaml, datetime
  out = Path(__file__).parent / 'results'; out.mkdir(parents=True, exist_ok=True)
  yaml.safe_dump({
    'model': 'dphubert_gated_effnet', 'target_params': args.target_params,
    'pruned_params': int(pruned_params), 'base_width': args.base_width,
    'feature_distill': args.feature_distill, 'seed': args.seed,
    'val_acc': round(best['acc'], 4), 'val_auc': round(best['auc'], 4),
    'acc_after_prune_nofinetune': round(acc0, 4),
    'finished_utc': datetime.datetime.utcnow().isoformat(),
  }, open(out / 'dphubert_{}p_{}_s{}.yaml'.format(args.target_params, args.feature_distill, args.seed), 'w'), sort_keys=False)


if __name__ == '__main__':
  main()
