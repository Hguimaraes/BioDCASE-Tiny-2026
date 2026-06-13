# --
# Hard Concrete distribution for differentiable L0 / structured pruning.
#
# Ported (lightly) from DPHuBERT's wav2vec2/hardconcrete.py, which is in turn
# from FLOP (https://github.com/asappresearch/flop/blob/master/flop/hardconcrete.py).
# Each gate carries n_in learnable log_alpha; l0_norm() is the expected number of
# OPEN gates (differentiable), used by the size-targeting Lagrangian. forward()
# samples a stochastic mask in train and a compiled deterministic mask in eval.

import math
import torch
import torch.nn as nn


class HardConcrete(nn.Module):
  def __init__(self, n_in, init_mean=0.01, init_std=0.01, temperature=2 / 3,
               stretch=0.1, eps=1e-6):
    super().__init__()
    self.n_in = n_in
    self.limit_l, self.limit_r = -stretch, 1.0 + stretch
    self.log_alpha = nn.Parameter(torch.zeros(n_in))
    self.beta = temperature
    self.init_mean, self.init_std = init_mean, init_std
    self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)
    self.eps = eps
    self.compiled_mask = None
    self.reset_parameters()

  def reset_parameters(self):
    self.compiled_mask = None
    mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
    self.log_alpha.data.normal_(mean, self.init_std)

  def l0_norm(self):
    """Expected number of open gates (differentiable in log_alpha)."""
    return (self.log_alpha + self.bias).sigmoid().sum()

  def forward(self):
    if self.training:
      self.compiled_mask = None
      u = self.log_alpha.new(self.n_in).uniform_(self.eps, 1 - self.eps)
      s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
      s = s * (self.limit_r - self.limit_l) + self.limit_l
      return s.clamp(min=0.0, max=1.0)
    # eval: deterministic compiled mask (zero out the expected-#-closed smallest)
    if self.compiled_mask is None:
      expected_num_zeros = self.n_in - self.l0_norm().item()
      num_zeros = round(expected_num_zeros)
      soft = torch.sigmoid(self.log_alpha / self.beta * 0.8)
      if num_zeros > 0:
        _, idx = torch.topk(soft, k=min(num_zeros, self.n_in), largest=False)
        soft[idx] = 0.0
      self.compiled_mask = soft
    return self.compiled_mask

  def keep_index(self):
    """Indices of gates kept open in the compiled (eval) mask -> for physical prune."""
    self.eval()
    mask = self.forward()
    return (mask > 0).nonzero(as_tuple=True)[0]

  def extra_repr(self):
    return str(self.n_in)


if __name__ == '__main__':
  torch.manual_seed(0)
  g = HardConcrete(64, init_mean=0.5)
  print('init l0_norm (~half open):', float(g.l0_norm()))
  g.train(); m = g()
  print('train mask shape', tuple(m.shape), 'in [0,1]:', float(m.min()), float(m.max()))
  g.eval(); print('eval kept gates:', len(g.keep_index()), '/', g.n_in)
  # gradient flows to log_alpha through l0_norm
  loss = (g.l0_norm() - 10.0) ** 2; loss.backward()
  print('grad on log_alpha finite:', bool(torch.isfinite(g.log_alpha.grad).all()))
