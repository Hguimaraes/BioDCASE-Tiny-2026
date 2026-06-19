# --
# Agreement/confidence filtering of the extra clips, decoupled from the Perch pass.
#
# The expensive Perch encode happens ONCE (export_embeddings.py -> embeddings.npz).
# This tool turns those embeddings into per-clip teacher LOGITS (apply the trained
# 11-class head) and then lets you choose the confidence threshold *afterwards*:
#
#   predict : embeddings.npz + teacher head  -> predictions.npz (stems, folder, logits)
#   sweep   : predictions.npz                -> per-class survivor counts over a tau grid
#   select  : predictions.npz --threshold T  -> manifest of kept clip paths (+ counts)
#
# Filter criteria (clips are labeled by Perch, the folder is the weak prior):
#   agree_conf  : argmax(teacher) == folder AND max softmax prob >= tau   (default)
#   folder_prob : softmax prob assigned to the folder class >= tau        (softer)
#
# predict needs torch + the head (.venv); sweep/select are pure numpy. Because
# logits are stored, you re-threshold instantly -- no Perch re-run.

import sys
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def _softmax(z):
  z = z - z.max(1, keepdims=True)
  e = np.exp(z)
  return e / e.sum(1, keepdims=True)


def cmd_predict(args):
  import torch
  from experiments.perch.train_teacher_head import build_head
  d = np.load(args.emb, allow_pickle=True)
  stems = np.array([str(s) for s in d['stems']])
  emb = d['embeddings'].astype(np.float32)
  folder_names = [str(n) for n in d['label_names']]
  folder = np.array([folder_names[int(i)] for i in d['labels']])

  ckpt = torch.load(args.head, map_location='cpu', weights_only=False)
  head_names = [str(n) for n in ckpt['label_names']]
  head = build_head(ckpt['head'], emb.shape[1], len(head_names), ckpt['hidden'], ckpt['dropout'])
  head.load_state_dict(ckpt['state_dict']); head.eval()
  x = (emb - ckpt['mu']) / ckpt['sd']                      # same standardization as training

  logits = []
  with torch.no_grad():
    for i in range(0, len(x), args.batch):
      logits.append(head(torch.from_numpy(x[i:i + args.batch].astype(np.float32))).numpy())
  logits = np.concatenate(logits).astype(np.float32)

  np.savez_compressed(args.out, stems=stems, folder=folder, logits=logits, head_names=np.array(head_names))
  print('predictions -> {} | {} clips, {} classes'.format(args.out, len(stems), len(head_names)))


def _derive(pred):
  logits = pred['logits']
  head_names = [str(n) for n in pred['head_names']]
  folder = np.array([str(s) for s in pred['folder']])
  prob = _softmax(logits)
  pred_name = np.array([head_names[i] for i in prob.argmax(1)])
  conf = prob.max(1)
  name_to_head = {n: i for i, n in enumerate(head_names)}
  folder_prob = np.array([prob[i, name_to_head[f]] if f in name_to_head else 0.0 for i, f in enumerate(folder)])
  return folder, pred_name, conf, folder_prob


def _mask_at(folder, pred_name, conf, folder_prob, criterion, tau):
  if criterion == 'agree_conf':
    return (pred_name == folder) & (conf >= tau)
  return folder_prob >= tau                                # folder_prob: argmax-agnostic


def cmd_sweep(args):
  pred = np.load(args.pred, allow_pickle=True)
  folder, pred_name, conf, folder_prob = _derive(pred)
  classes = sorted(set(folder.tolist()))
  print('criterion: {} | {} clips'.format(args.criterion, len(folder)))
  print('{:<26}'.format('class') + ''.join('{:>9}'.format('t>=%.2f' % t) for t in args.taus))
  for c in classes:
    ci = folder == c
    row = [int((_mask_at(folder, pred_name, conf, folder_prob, args.criterion, t) & ci).sum()) for t in args.taus]
    print('{:<26}'.format(c) + ''.join('{:>9,}'.format(r) for r in row))
  tot = [int(_mask_at(folder, pred_name, conf, folder_prob, args.criterion, t).sum()) for t in args.taus]
  print('{:<26}'.format('TOTAL') + ''.join('{:>9,}'.format(t) for t in tot))


def cmd_select(args):
  pred = np.load(args.pred, allow_pickle=True)
  folder, pred_name, conf, folder_prob = _derive(pred)
  stems = np.array([str(s) for s in pred['stems']])
  mask = _mask_at(folder, pred_name, conf, folder_prob, args.criterion, args.threshold)
  root = Path(args.clips_root)
  lines = [str(root / folder[i] / (stems[i] + '.wav')) for i in np.where(mask)[0]]
  Path(args.out).write_text('\n'.join(lines) + ('\n' if lines else ''))
  print('selected {}/{} clips @ {} {} -> {}'.format(mask.sum(), len(mask), args.criterion, args.threshold, args.out))
  for c in sorted(set(folder.tolist())):
    print('  {:<26} {}'.format(c, int((mask & (folder == c)).sum())))


def main():
  ap = argparse.ArgumentParser(description='Agreement/confidence filtering of extra clips.')
  sub = ap.add_subparsers(dest='cmd', required=True)

  p = sub.add_parser('predict', help='embeddings + head -> predictions.npz')
  p.add_argument('--emb', required=True); p.add_argument('--head', required=True)
  p.add_argument('--out', default='predictions.npz'); p.add_argument('--batch', type=int, default=4096)
  p.set_defaults(func=cmd_predict)

  s = sub.add_parser('sweep', help='per-class survivor counts over a tau grid')
  s.add_argument('--pred', required=True)
  s.add_argument('--criterion', choices=['agree_conf', 'folder_prob'], default='agree_conf')
  s.add_argument('--taus', type=float, nargs='+', default=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
  s.set_defaults(func=cmd_sweep)

  e = sub.add_parser('select', help='write manifest of kept clips at a threshold')
  e.add_argument('--pred', required=True); e.add_argument('--threshold', type=float, required=True)
  e.add_argument('--criterion', choices=['agree_conf', 'folder_prob'], default='agree_conf')
  e.add_argument('--clips-root', required=True, help='root holding <class>/<stem>.wav')
  e.add_argument('--out', default='kept_clips.txt')
  e.set_defaults(func=cmd_select)

  args = ap.parse_args()
  args.func(args)


if __name__ == '__main__':
  main()
