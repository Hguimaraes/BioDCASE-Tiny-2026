# --
# logit-level knowledge distillation from the Perch v2 teacher (track C1)
#
# the teacher's raw 11-class logits are exported once per clip (by
# experiments/perch/train_teacher_head.py), keyed by wav stem. here we
# align them to the student's training samples via the datamodule's
# sid -> stem mapping, and combine a KL distillation term with the
# standard cross-entropy on the hard labels. mixup is applied consistently
# to inputs, hard targets and teacher probabilities so the two losses stay
# aligned with the mixed input.

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]


def load_teacher_logits(teacher_dir, split):
  """
  load exported teacher soft logits for a split -> {stem: logits (num_classes,)}
  """

  d = np.load(Path(teacher_dir) / 'soft_logits_{}.npz'.format(split), allow_pickle=True)
  stems, logits = d['stems'], d['logits'].astype(np.float32)
  return {str(s): logits[i] for i, s in enumerate(stems)}


def load_msab(msab_dir, split):
  """
  load exported MSAB context vectors for a split -> {stem: msab (msab_dim,)}
  """

  d = np.load(Path(msab_dir) / '{}.npz'.format(split), allow_pickle=True)
  stems, msab = d['stems'], d['msab'].astype(np.float32)
  return {str(s): msab[i] for i, s in enumerate(stems)}


def _sid_to_array(datamodule, stem_to_array, what):
  """
  map each sample id to its per-clip array via the datamodule's sid->stem map;
  fail loudly on any missing stem
  """

  out, missing = {}, 0
  for sid in [int(s) for s in datamodule.sample_ids]:
    stem = datamodule.get_file_name_id_by_single_sid(sid)
    a = stem_to_array.get(stem)
    if a is None: missing += 1
    else: out[sid] = a
  if missing: raise ValueError('{}: {} samples have no entry (stem mismatch)'.format(what, missing))
  return out


class ContextDataset(torch.utils.data.Dataset):
  """
  wraps a feature dataset and attaches the MSAB context vector per sample,
  looked up by stem. returns (mel, y, sid, ctx). used for validation/test of
  FiLM models (no teacher logits needed there).
  """

  def __init__(self, base_dataset, datamodule, stem_to_ctx):
    super().__init__()
    self.base = base_dataset
    self.sid_to_ctx = _sid_to_array(datamodule, stem_to_ctx, 'ContextDataset')

  def __len__(self):
    return len(self.base)

  def __getitem__(self, idx):
    x, y, sid = self.base[idx]
    ctx = torch.from_numpy(self.sid_to_ctx[int(sid)]).float()
    return x, y, sid, ctx


class TeacherLogitDataset(torch.utils.data.Dataset):
  """
  wraps the (augmented) student dataset and attaches the teacher's logits
  for each sample, looked up by stem via the datamodule's sid -> stem map
  """

  def __init__(self, base_dataset, datamodule, stem_to_logits, num_classes, stem_to_ctx=None):

    super().__init__()
    self.base = base_dataset
    self.num_classes = num_classes

    # precompute sid -> teacher logits, fail loudly on any miss
    self.sid_to_logits = _sid_to_array(datamodule, stem_to_logits, 'TeacherLogitDataset')
    # optional sid -> MSAB context (for FiLM students, track C2)
    self.sid_to_ctx = _sid_to_array(datamodule, stem_to_ctx, 'TeacherLogitDataset(ctx)') if stem_to_ctx is not None else None


  def __len__(self):
    return len(self.base)


  def __getitem__(self, idx):
    x, y, sid = self.base[idx]
    t_logits = torch.from_numpy(self.sid_to_logits[int(sid)]).float()
    if self.sid_to_ctx is None:
      return x, y, sid, t_logits
    ctx = torch.from_numpy(self.sid_to_ctx[int(sid)]).float()
    return x, y, sid, t_logits, ctx


def mixup_with_teacher(x, y, t_logits, num_classes, alpha=0.2, p=0.5):
  """
  batch-level mixup applied jointly to inputs, one-hot hard targets and
  teacher logits. returns (x, y_soft, t_logits) all mixed with the same
  (lambda, permutation).
  """

  y_soft = F.one_hot(y.to(torch.int64), num_classes=num_classes).float()

  if alpha <= 0 or torch.rand(1).item() >= p:
    return x, y_soft, t_logits

  lam = float(np.random.beta(alpha, alpha))
  perm = torch.randperm(x.shape[0])
  x = lam * x + (1.0 - lam) * x[perm]
  y_soft = lam * y_soft + (1.0 - lam) * y_soft[perm]
  t_logits = lam * t_logits + (1.0 - lam) * t_logits[perm]
  return x, y_soft, t_logits


def distillation_loss(student_logits, teacher_logits, hard_soft_targets, alpha=0.5, temperature=3.0, label_smoothing=0.0):
  """
  combined loss: alpha * KD(KL, temperature) + (1 - alpha) * CE(hard).

  - KD term: T^2 * KL( log_softmax(student/T) || softmax(teacher/T) )
  - CE term: cross-entropy against (possibly mixed) soft hard-label targets,
    with optional label smoothing folded into those targets.
  """

  T = temperature

  # distillation term (teacher probabilities at temperature)
  teacher_prob = F.softmax(teacher_logits / T, dim=1)
  student_logT = F.log_softmax(student_logits / T, dim=1)
  kd = F.kl_div(student_logT, teacher_prob, reduction='batchmean') * (T * T)

  # hard cross-entropy term against soft targets (supports mixup + smoothing)
  if label_smoothing > 0:
    n = hard_soft_targets.shape[1]
    hard_soft_targets = hard_soft_targets * (1.0 - label_smoothing) + label_smoothing / n
  ce = -(hard_soft_targets * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()

  loss = alpha * kd + (1.0 - alpha) * ce
  return loss, {'kd': float(kd.detach()), 'ce': float(ce.detach())}


if __name__ == '__main__':
  """ smoke test """

  B, K = 8, 11
  student = torch.randn(B, K, requires_grad=True)
  teacher = torch.randn(B, K)
  y = torch.randint(0, K, (B,))
  x = torch.randn(B, 1, 40, 133)
  xm, ys, tm = mixup_with_teacher(x, y, teacher, K, alpha=0.2, p=1.0)
  print('mixup shapes:', xm.shape, ys.shape, tm.shape, '| y_soft rows sum:', ys.sum(1)[:3].tolist())
  loss, parts = distillation_loss(student, teacher, F.one_hot(y, K).float(), alpha=0.5, temperature=3.0, label_smoothing=0.1)
  loss.backward()
  print('loss:', round(float(loss), 4), parts, '| grad ok:', student.grad is not None)
