# --
# Gated EfficientNet-style CNN for DPHuBERT-style structured pruning.
#
# MBConv blocks with a HardConcrete gate on the EXPAND (intermediate) channels
# only -- block input/output channels stay fixed, so residual connections are
# never broken (the safe, high-leverage prune dimension; EfficientNet's expand
# 1x1 convs hold most of the params). get_num_params() is DIFFERENTIABLE in the
# gates (uses gate.l0_norm()) so the size-targeting Lagrangian can drive the
# model to an exact param budget. prune() rebuilds a compact, gate-free model.
#
# Input: 40-mel PCEN (1,40,133). Deployable student for Perch distillation.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.pruning.hardconcrete import HardConcrete

# EfficientNet base stages: (expand_ratio, out_ch, repeats, stride, kernel)
BASE_STAGES = [
  (1, 16, 1, 1, 3), (6, 24, 2, 2, 3), (6, 40, 2, 2, 5),
  (6, 80, 3, 2, 3), (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3),
]


def _round_ch(c, w, divisor=8):
  c = c * w
  new = max(divisor, int(c + divisor / 2) // divisor * divisor)
  if new < 0.9 * c:
    new += divisor
  return int(new)


def _round_rep(r, d):
  return int(math.ceil(d * r))


class GatedMBConv(nn.Module):
  def __init__(self, in_ch, out_ch, expand_ch, kernel, stride, se_ratio=0.25, use_gate=True):
    super().__init__()
    self.in_ch, self.out_ch, self.expand_ch = in_ch, out_ch, expand_ch
    self.use_residual = (stride == 1 and in_ch == out_ch)
    self.has_expand = expand_ch != in_ch
    pad = kernel // 2

    if self.has_expand:
      self.expand = nn.Conv2d(in_ch, expand_ch, 1, bias=False)
      self.bn0 = nn.BatchNorm2d(expand_ch)
    self.dw = nn.Conv2d(expand_ch, expand_ch, kernel, stride, pad, groups=expand_ch, bias=False)
    self.bn1 = nn.BatchNorm2d(expand_ch)
    se_mid = max(1, int(in_ch * se_ratio))
    self.se_reduce = nn.Conv2d(expand_ch, se_mid, 1)
    self.se_expand = nn.Conv2d(se_mid, expand_ch, 1)
    self.project = nn.Conv2d(expand_ch, out_ch, 1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_ch)
    self.act = nn.SiLU()
    self.gate = HardConcrete(expand_ch) if use_gate else None

  def forward(self, x):
    inp = x
    if self.has_expand:
      x = self.act(self.bn0(self.expand(x)))
    x = self.act(self.bn1(self.dw(x)))
    if self.gate is not None:                       # mask expand channels
      x = x * self.gate().view(1, -1, 1, 1)
    se = x.mean((2, 3), keepdim=True)
    se = torch.sigmoid(self.se_expand(self.act(self.se_reduce(se))))
    x = self.bn2(self.project(x * se))
    if self.use_residual:
      x = x + inp
    return x

  def expected_expand(self):
    return self.gate.l0_norm() if self.gate is not None else torch.tensor(float(self.expand_ch))

  def get_num_params(self):
    """Differentiable expected param count given the expand gate's L0."""
    e = self.expected_expand()
    se_mid = self.se_reduce.out_channels
    p = 0.0
    if self.has_expand:
      p = p + self.in_ch * e + 2 * e          # expand conv + bn0
    p = p + e * (self.dw.kernel_size[0] ** 2) + 2 * e     # dw + bn1
    p = p + (e * se_mid + se_mid) + (se_mid * e + e)      # SE reduce + expand (with bias)
    p = p + e * self.out_ch + 2 * self.out_ch            # project + bn2
    return p

  @torch.no_grad()
  def to_pruned(self):
    """Build a compact gate-free GatedMBConv from the kept expand channels.

    The eval gate yields a SOFT mask (kept channels scaled by <1) applied after
    the SiLU, so we fold the kept scales into the two consumers of the masked
    activation -- SE-reduce and project input weights -- to reproduce it exactly.
    """
    if self.gate is not None:
      self.gate.eval()
      mask = self.gate()
      keep = (mask > 0).nonzero(as_tuple=True)[0]
      scale = mask[keep]
    else:
      keep = torch.arange(self.expand_ch); scale = torch.ones(self.expand_ch)
    k = len(keep)
    m = GatedMBConv(self.in_ch, self.out_ch, k, self.dw.kernel_size[0],
                    self.dw.stride[0], use_gate=False)
    if self.has_expand:
      m.expand.weight.copy_(self.expand.weight[keep])
      _copy_bn(m.bn0, self.bn0, keep)
    m.dw.weight.copy_(self.dw.weight[keep]); _copy_bn(m.bn1, self.bn1, keep)
    sc = scale.view(1, -1, 1, 1)
    m.se_reduce.weight.copy_(self.se_reduce.weight[:, keep] * sc)   # fold scale into inputs
    m.se_reduce.bias.copy_(self.se_reduce.bias)
    m.se_expand.weight.copy_(self.se_expand.weight[keep]); m.se_expand.bias.copy_(self.se_expand.bias[keep])
    m.project.weight.copy_(self.project.weight[:, keep] * sc)       # fold scale into inputs
    _copy_bn(m.bn2, self.bn2, torch.arange(self.out_ch))
    return m


def _copy_bn(dst, src, idx):
  dst.weight.copy_(src.weight[idx]); dst.bias.copy_(src.bias[idx])
  dst.running_mean.copy_(src.running_mean[idx]); dst.running_var.copy_(src.running_var[idx])


class GatedEffNet(nn.Module):
  def __init__(self, num_classes=11, in_ch=1, width=0.6, depth=1.0, head_dim=64,
               dropout=0.2, use_gate=True):
    super().__init__()
    stem = _round_ch(32, width)
    self.stem = nn.Sequential(nn.Conv2d(in_ch, stem, 3, 2, 1, bias=False),
                              nn.BatchNorm2d(stem), nn.SiLU())
    blocks, c = [], stem
    for er, oc, rep, st, k in BASE_STAGES:
      oc = _round_ch(oc, width)
      for i in range(_round_rep(rep, depth)):
        stride = st if i == 0 else 1
        exp = c * er
        blocks.append(GatedMBConv(c, oc, exp, k, stride, use_gate=use_gate))
        c = oc
    self.blocks = nn.ModuleList(blocks)
    head = _round_ch(1280, width)
    self.head = nn.Sequential(nn.Conv2d(c, head, 1, bias=False), nn.BatchNorm2d(head), nn.SiLU())
    self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(head, head_dim),
                                    nn.SiLU(), nn.Linear(head_dim, num_classes))
    self.head_ch = head

  def forward_features(self, x):
    x = self.stem(x)
    for b in self.blocks:
      x = b(x)
    return self.head(x)                              # (B, head, H, W) pre-pool map

  def forward(self, x):
    feat = self.forward_features(x)
    return self.classifier(feat.mean((2, 3)))

  def forward_with_spatial(self, x):
    feat = self.forward_features(x)
    return self.classifier(feat.mean((2, 3))), feat

  def get_num_params(self):
    """Differentiable expected total params (gated parts) + fixed parts."""
    fixed = sum(p.numel() for n, p in self.named_parameters()
                if not any(g in n for g in ['blocks.'])) \
        - sum(g.log_alpha.numel() for g in self._gates())   # log_alpha aren't model params
    block = sum(b.get_num_params() for b in self.blocks)
    return block + fixed

  def _gates(self):
    return [b.gate for b in self.blocks if b.gate is not None]

  @torch.no_grad()
  def to_pruned(self):
    self.eval()
    pruned = GatedEffNet.__new__(GatedEffNet)
    nn.Module.__init__(pruned)
    pruned.stem = self.stem
    pruned.blocks = nn.ModuleList([b.to_pruned() for b in self.blocks])
    pruned.head = self.head
    pruned.classifier = self.classifier
    pruned.head_ch = self.head_ch
    return pruned


if __name__ == '__main__':
  torch.manual_seed(0)
  m = GatedEffNet(width=0.6)
  x = torch.zeros(2, 1, 40, 133)
  logits, feat = m.forward_with_spatial(x)
  actual = sum(p.numel() for n, p in m.named_parameters() if 'log_alpha' not in n)
  print('forward ok | logits', tuple(logits.shape), '| spatial', tuple(feat.shape))
  print('actual params (full):', actual)
  print('get_num_params (gates ~all open):', round(float(m.get_num_params())))
