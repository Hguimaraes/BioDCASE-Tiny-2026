# --
# model tiny ml

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

# add root path of project if called as main
if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent)] if p not in sys.path]

from pipeline_pytorch.model_base import ModelBase


class Baseline(ModelBase):
  """
  model tiny ml - overwrite model base
  """

  def define_network_structure(self, n_filters = 32, dropout=0.05):

    assert len(self.cfg['input_shape']) == 3

    # Feature extractor
    self.features = nn.Sequential(
        nn.Conv2d(self.cfg['input_shape'][0], n_filters, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(n_filters, n_filters * 2, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(4),

        nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
    )

    # Classifier
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(n_filters * 4, 32),
        nn.ReLU(),
        nn.Linear(32, self.cfg['num_classes'])
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

  def forward_with_features(self, x):
    """
    forward that also returns the penultimate GAP descriptor (B, n_filters*4).
    used only at train time for embedding/feature distillation (the hint head
    is external to the model, so forward() and the exported graph are unchanged).
    """
    f = self.features(x)            # (B, C, 1, 1) after global avg pool
    logits = self.classifier(f)     # classifier starts with Flatten
    feat = torch.flatten(f, 1)      # (B, C) embedding for the hint loss
    return logits, feat


class ConformerStudent(ModelBase):
  """Small Conformer student (torchaudio) over the PCEN map: the 40 mel bins are
  projected to d_model and the 133 frames form the sequence. The time-pooled
  d_model descriptor is the representation the embedding-distillation lever
  regresses onto Perch. Single-input drop-in for run_model_training.
  cfg: d_model (64), num_layers (4), num_heads (4), ffn_dim (128),
  conv_kernel (15, odd), dropout (0.1)."""

  def define_network_structure(self):
    from torchaudio.models import Conformer                  # lazy: keep module importable w/o torchaudio
    _, mel, _ = self.cfg['input_shape']                      # (1, 40, 133)
    d = self.cfg.get('d_model', 64)
    self.in_proj = nn.Linear(mel, d)
    self.conformer = Conformer(input_dim=d,
                               num_heads=self.cfg.get('num_heads', 4),
                               ffn_dim=self.cfg.get('ffn_dim', 128),
                               num_layers=self.cfg.get('num_layers', 4),
                               depthwise_conv_kernel_size=self.cfg.get('conv_kernel', 15),
                               dropout=self.cfg.get('dropout', 0.1))
    self.gap_dim = d
    self.classifier = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, self.cfg['num_classes']))

  def _features(self, x):
    z = x[:, 0].transpose(1, 2)                              # (B, 1, 40, 133) -> (B, 133, 40)
    z = self.in_proj(z)                                      # (B, 133, d)
    lengths = torch.full((z.shape[0],), z.shape[1], device=z.device, dtype=torch.long)
    z, _ = self.conformer(z, lengths)                        # (B, 133, d)
    return z.mean(dim=1)                                     # (B, d) time-pooled descriptor

  def forward(self, x):
    return self.classifier(self._features(x))

  def forward_with_features(self, x):
    feat = self._features(x)
    return self.classifier(feat), feat                       # embed distill regresses feat

  def save_model_to_tflite(self):
    print("\n*** ConformerStudent: tflite export deferred (attention/conv in graph). "
          "float .pth metrics are logged.")
    return


def _gn_groups(c):
  """largest of (8,4,2,1) dividing c -> GroupNorm groups (no running buffers, so
  unlike BatchNorm it's identical at train/eval and safe under WeightEMA)."""
  return next(k for k in (8, 4, 2, 1) if c % k == 0)


class GaborSTRFConv(nn.Module):
  """First conv as a 2-D Gabor spectro-temporal modulation filterbank (auditory
  STRF, Chi-Ru-Shamma): out_ch filters over a grid of (spectral scale, temporal
  rate). Gabor-initialized, learnable. A plain Conv2d, so fully deployable."""
  def __init__(self, out_ch=32, kf=9, kt=9, learnable=True):
    super().__init__()
    self.conv = nn.Conv2d(1, out_ch, (kf, kt), padding=(kf // 2, kt // 2), bias=False)
    fa = torch.arange(kf).float() - kf // 2; ta = torch.arange(kt).float() - kt // 2
    F2, T2 = torch.meshgrid(fa, ta, indexing='ij')
    win = torch.exp(-(F2 ** 2 + T2 ** 2) / (2 * (kf / 4.0) ** 2))
    g = int(out_ch ** 0.5) + 1; rn = (out_ch + g - 1) // g
    ker = [win * torch.cos(2 * math.pi * (s * F2 + r * T2))
           for s in torch.linspace(0.1, 0.5, g) for r in torch.linspace(0.1, 0.5, rn)][:out_ch]
    ker = torch.stack(ker)
    ker = ker - ker.mean(dim=(1, 2), keepdim=True)
    ker = ker / (ker.flatten(1).norm(dim=1)[:, None, None] + 1e-8)
    with torch.no_grad():
      self.conv.weight.copy_(ker.unsqueeze(1))
    self.conv.weight.requires_grad_(learnable)

  def forward(self, x):
    return self.conv(x)


class LearnablePCEN(nn.Module):
  """Per-Channel Energy Normalization with PER-BAND LEARNABLE params, init at the
  Perch bioacoustic defaults (smoothing 0.145, gain 0.8, bias 10, root 4). Operates
  on raw magnitude mel ENERGY (B, 1, M, T) -> PCEN-mel (B, 1, M, T). The causal EMA
  over time is the same recursion already shipped offline (tflite-safe; at deploy
  the learned params can be frozen back into the offline front-end). Params are kept
  in valid ranges via sigmoid (s in (0,1)) / softplus (gain,bias,root > 0) so the
  AGC stays well-defined. Only Parameters, no running buffers -> WeightEMA-safe.
  Wang et al. 'Trainable Frontend for Robust and Far-Field Keyword Spotting'."""
  def __init__(self, n_mels=40, smoothing=0.145, gain=0.8, bias=10.0, root=4.0, eps=1e-6):
    super().__init__()
    self.eps = eps
    inv_sig = lambda y: math.log(y / (1.0 - y))                                 # sigmoid^-1
    inv_sp = lambda y: math.log(math.expm1(y))                                  # softplus^-1
    self.s_ = nn.Parameter(torch.full((n_mels,), inv_sig(smoothing)))          # smoothing coef
    self.a_ = nn.Parameter(torch.full((n_mels,), inv_sp(gain)))                # gain (alpha)
    self.d_ = nn.Parameter(torch.full((n_mels,), inv_sp(bias)))                # bias (delta)
    self.r_ = nn.Parameter(torch.full((n_mels,), inv_sp(root)))                # root (r)

  def forward(self, x):                                                        # (B, 1, M, T) raw mel energy
    e = x.squeeze(1)                                                           # (B, M, T)
    M, T = e.shape[-2], e.shape[-1]
    s = torch.sigmoid(self.s_).view(1, M)                                      # per-band, in (0,1)
    a = F.softplus(self.a_).view(1, M, 1)
    d = F.softplus(self.d_).view(1, M, 1)
    inv_r = 1.0 / F.softplus(self.r_).view(1, M, 1)
    prev = e[..., 0]                                                           # causal EMA along time, per band
    cols = [prev]                                                              # (no in-place writes -> autograd-safe)
    for t in range(1, T):
      prev = (1.0 - s) * prev + s * e[..., t]
      cols.append(prev)
    m = torch.stack(cols, dim=-1)                                             # (B, M, T)
    out = (e / (self.eps + m) ** a + d) ** inv_r - d ** inv_r                  # PCEN
    return out.unsqueeze(1)                                                    # (B, 1, M, T)


class LearnFrontNet(ModelBase):
  """The proven `Baseline` tiny CNN fed by a LEARNABLE PCEN front-end (LearnablePCEN,
  per-band gain/bias/root/smoothing) operating on the raw magnitude mel-energy cache
  (cache_mel), not the fixed-PCEN cache. At init it reproduces the offline fixed-PCEN
  feature (same params + the same per-clip min-max), then the front-end adapts jointly
  with the net + D2 distillation. This is the one A/B on the proven PCEN lever: fixed
  vs learned compression. Body/GAP/classifier are Baseline-identical, so the embed-
  distill lever and exported graph are unchanged. Deployable (PCEN recursion + convs).
  cfg: n_filters (32), n_mels (40), pcen_init (dict), dropout (0.05)."""

  def define_network_structure(self):
    nf = self.cfg.get('n_filters', 32)
    self.front = LearnablePCEN(n_mels=self.cfg.get('n_mels', 40), **(self.cfg.get('pcen_init') or {}))
    self.features = nn.Sequential(
        nn.Conv2d(1, nf, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(nf, nf * 2, kernel_size=3), nn.ReLU(), nn.MaxPool2d(4),
        nn.Conv2d(nf * 2, nf * 4, kernel_size=3), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)))
    self.classifier = nn.Sequential(
        nn.Flatten(), nn.Dropout(self.cfg.get('dropout', 0.05)),
        nn.Linear(nf * 4, 32), nn.ReLU(), nn.Linear(32, self.cfg['num_classes']))

  def _front(self, x):
    z = self.front(x)                                                          # (B, 1, M, T) PCEN-mel
    zf = z.flatten(1)                                                          # per-clip min-max to [0,1]
    lo = zf.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)                      # (matches offline FeatureHandler
    hi = zf.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)                      #  normalize_features=True)
    return (z - lo) / (hi - lo + 1e-6)

  def forward(self, x):
    return self.classifier(self.features(self._front(x)))

  def forward_with_features(self, x):
    f = self.features(self._front(x))
    return self.classifier(f), torch.flatten(f, 1)                            # embed distill on the GAP descriptor


class StrfBaseline(ModelBase):
  """The proven `Baseline` tiny CNN with ONE change: the first 3x3 conv is replaced
  by a 2-D Gabor spectro-temporal receptive-field filterbank (GaborSTRFConv). The
  STRF is the single signal-processing bias that helped on this task (best AUC at
  ~1x Baseline params); the rest of the body, the GAP descriptor, and the D2 distill
  recipe are unchanged, so it's a ~drop-in for Baseline and exports to tflite (plain
  Conv2d throughout -- no FFT, no grouped/attention machinery).
  cfg: n_filters (32), strf_kernel (9, odd), learnable_strf (True), dropout (0.05)."""

  def define_network_structure(self):
    assert len(self.cfg['input_shape']) == 3
    nf = self.cfg.get('n_filters', 32)
    k = self.cfg.get('strf_kernel', 9)
    learn = self.cfg.get('learnable_strf', True)

    # Feature extractor: Baseline body, but the first conv is the Gabor STRF front
    self.features = nn.Sequential(
        GaborSTRFConv(nf, kf=k, kt=k, learnable=learn),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(nf, nf * 2, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(4),

        nn.Conv2d(nf * 2, nf * 4, kernel_size=3),
        nn.ReLU(),

        nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
    )

    # Classifier (identical to Baseline)
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(self.cfg.get('dropout', 0.05)),
        nn.Linear(nf * 4, 32),
        nn.ReLU(),
        nn.Linear(32, self.cfg['num_classes']),
    )

  def forward(self, x):
    return self.classifier(self.features(x))

  def forward_with_features(self, x):
    """forward that also returns the penultimate GAP descriptor (B, n_filters*4)
    for embedding/feature distillation -- same contract as Baseline, so the D2
    embed-distill lever is unchanged and the exported graph is forward()."""
    f = self.features(x)            # (B, C, 1, 1) after global avg pool
    logits = self.classifier(f)
    feat = torch.flatten(f, 1)      # (B, C) embedding for the hint loss
    return logits, feat


class _TemporalBlock(nn.Module):
  """Simple dilated 1-D residual block over time (NO branching/gating): depthwise
  dilated Conv1d -> pointwise Conv1d -> GroupNorm -> ReLU, residual. Length is
  preserved (pad = (k//2)*dilation). Depthwise-separable to stay compact; GroupNorm
  (no running stats) is EMA-safe. Deployable (conv/groupnorm/relu)."""
  def __init__(self, ch, kernel=7, dilation=1):
    super().__init__()
    pad = (kernel // 2) * dilation
    self.dw = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation, groups=ch, bias=False)
    self.pw = nn.Conv1d(ch, ch, 1)
    self.gn = nn.GroupNorm(_gn_groups(ch), ch)

  def forward(self, x):                                                        # (B, ch, T) -> (B, ch, T)
    y = torch.relu(self.gn(self.pw(self.dw(x))))
    return x + y


class FreqTimeNet(ModelBase):
  """Frequency/time-SEPARATED tiny CNN (Tan et al. 2019 GRN, adapted to classification).
  A FREQUENCY module of 2-D convs (GaborSTRF front + freq-dilated 3x3) collapses the mel
  axis while KEEPING the full 133-frame time axis; a TIME module of simple dilated 1-D
  conv residual blocks models temporal structure (rhythm); temporal mean+std pooling ->
  classifier. The pooled descriptor is the embed-distill target. Motivation: handle
  frequency and time separately (Tan: vertical=timbre, horizontal=temporal evolution;
  time axis >> freq axis). MINIMAL first step -- NO branching/GLU/FiLM/Fourier (deferred).
  cfg: n_filters (32), tcn_ch (64), tcn_kernel (7), tcn_dilations ([1,2,4,8]),
  strf (True), learnable_strf (True), dropout (0.05). Deployable (conv/groupnorm/linear)."""

  def define_network_structure(self):
    nf = self.cfg.get('n_filters', 32); ch = self.cfg.get('tcn_ch', 64)
    learn = self.cfg.get('learnable_strf', True)
    front = GaborSTRFConv(nf, learnable=learn) if self.cfg.get('strf', True) else nn.Conv2d(1, nf, 9, padding=4)
    self.freq = nn.Sequential(                                                # collapse FREQ, keep TIME (freq-dilated)
        front, nn.ReLU(), nn.MaxPool2d((2, 1)),
        nn.Conv2d(nf, ch, 3, dilation=(2, 1), padding=(2, 1)), nn.ReLU(), nn.MaxPool2d((2, 1)),
        nn.Conv2d(ch, ch, 3, dilation=(2, 1), padding=(2, 1)), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, None)))                                      # (B, ch, 1, T)
    self.time = nn.ModuleList([_TemporalBlock(ch, self.cfg.get('tcn_kernel', 7), d)
                               for d in self.cfg.get('tcn_dilations', [1, 2, 4, 8])])
    self.gap_dim = 2 * ch                                                     # mean ++ std
    self.classifier = nn.Sequential(nn.Dropout(self.cfg.get('dropout', 0.05)),
                                    nn.Linear(self.gap_dim, 32), nn.ReLU(),
                                    nn.Linear(32, self.cfg['num_classes']))

  def _features(self, x):
    z = self.freq(x).squeeze(2)                                              # (B, ch, T) freq collapsed, time kept
    for blk in self.time:
      z = blk(z)                                                             # dilated 1-D temporal modeling
    return torch.cat([z.mean(dim=2), z.std(dim=2)], dim=1)                   # temporal stats pooling (B, 2ch)

  def forward(self, x):
    return self.classifier(self._features(x))

  def forward_with_features(self, x):
    f = self._features(x)
    return self.classifier(f), f                                            # embed distill regresses the pooled vec

  def save_model_to_tflite(self):
    try:
      return super().save_model_to_tflite()
    except Exception as e:
      print("\n*** FreqTimeNet: tflite export deferred ({}). float .pth metrics are logged.".format(type(e).__name__))
      return


class BaselineGRU(ModelBase):
  """
  Track F CRNN: the exact Baseline conv stack followed by a GRU over time,
  i.e. the "MobileGRU" idea (Dhar, BioDCASE 2025) ported onto our best model.

  The only change vs Baseline is the head. Baseline collapses the conv map
  (C, F, T) with a global avg-pool over BOTH freq and time -> the temporal
  structure is discarded. Here we instead pool FREQUENCY only, leaving a
  length-T sequence of C-dim vectors, run a GRU over it, then temporal-pool ->
  dense. Conv layers (32->64->128), the PCEN front-end and the distillation
  recipe are unchanged, so this isolates the effect of recurrence.
  """

  def define_network_structure(self, n_filters=32):

    assert len(self.cfg['input_shape']) == 3

    gru_hidden = self.cfg.get('gru_hidden', 32)   # paper's GRU size; +13% params vs Baseline
    head_dim = self.cfg.get('head_dim', 32)
    dropout = self.cfg.get('dropout', 0.05)
    # we classify whole clips offline (not streaming), so the full sequence is
    # available -> a bidirectional GRU can use future context too. Config flag.
    bidirectional = self.cfg.get('bidirectional', False)

    # identical conv feature extractor to Baseline, MINUS the global pool so
    # the time axis survives into the GRU
    self.features = nn.Sequential(
        nn.Conv2d(self.cfg['input_shape'][0], n_filters, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(n_filters, n_filters * 2, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(4),

        nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=3),
        nn.ReLU(),
    )

    # collapse frequency only -> (B, C, 1, T); keep time as the GRU sequence
    self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
    self.gru = nn.GRU(input_size=n_filters * 4, hidden_size=gru_hidden,
                      batch_first=True, bidirectional=bidirectional)

    # bidirectional GRU emits 2*hidden per step (forward+backward concatenated)
    gru_out = gru_hidden * (2 if bidirectional else 1)

    # temporal global avg-pool over the GRU sequence, then classify
    self.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(gru_out, head_dim),
        nn.ReLU(),
        nn.Linear(head_dim, self.cfg['num_classes']),
    )

  def forward(self, x):
    x = self.features(x)                  # (B, C, F, T)
    x = self.freq_pool(x).squeeze(2)      # (B, C, T)
    x = x.permute(0, 2, 1)                # (B, T, C) sequence
    x, _ = self.gru(x)                    # (B, T, H)
    x = x.mean(dim=1)                     # temporal global average pool -> (B, H)
    return self.classifier(x)

  def save_model_to_tflite(self):
    # research-first (Track F): GRU int8 on esp-nn is non-trivial, and the
    # float export may not lower cleanly. Don't let an export failure crash a
    # training run -- the .pth metrics are what we measure. Build the deployable
    # path only if recurrence wins.
    try:
      return super().save_model_to_tflite()
    except Exception as e:
      print("\n*** BaselineGRU: tflite export deferred ({}). "
            "float .pth metrics are logged.".format(type(e).__name__))
      return


class FiLM2d(nn.Module):
  """
  Feature-wise Linear Modulation for conv feature maps (Track C2).
  Adapted from the BioME FiLM (Perez et al. 2018): a context vector produces
  per-channel scale (gamma) and shift (beta) applied as x' = gamma*x + beta,
  broadcast over the spatial dims. Lightweight side-channel conditioning.
  """

  def __init__(self, channels, context_dim):
    super().__init__()
    self.modulator = nn.Linear(context_dim, 2 * channels)

  def forward(self, x, ctx):
    # x: (B, C, H, W) ; ctx: (B, context_dim)
    gamma, beta = self.modulator(ctx).chunk(2, dim=-1)
    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    return gamma * x + beta


class DepthwiseSeparableBlock(nn.Module):
  """
  MobileNet-style depthwise-separable block: 3x3 depthwise conv (optionally
  strided) + 1x1 pointwise conv, each followed by BatchNorm + ReLU. All ops
  are int8 TFLite / esp-nn friendly (BN folds into the preceding conv).
  """

  def __init__(self, in_ch, out_ch, stride=1):
    super().__init__()
    self.block = nn.Sequential(
      # depthwise
      nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
      nn.BatchNorm2d(in_ch),
      nn.ReLU(),
      # pointwise
      nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
    )

  def forward(self, x):
    return self.block(x)


class SlimCNN(ModelBase):
  """
  Track B student: a depthwise-separable CNN on the baseline mel input
  (1 x mel x time). Replaces the baseline's dense 3x3 convs with cheaper
  DS blocks, freeing parameter budget for more depth/width. Architecture
  hyperparameters are read from config kwargs (defaults below) so width and
  depth can be swept.
  """

  def define_network_structure(self):

    assert len(self.cfg['input_shape']) == 3

    # arch hyperparameters (overridable via model kwargs in config)
    stem_ch = self.cfg.get('stem_ch', 24)
    block_widths = self.cfg.get('block_widths', [48, 64, 96, 128])
    block_strides = self.cfg.get('block_strides', [2, 2, 2, 1])
    head_dim = self.cfg.get('head_dim', 64)
    dropout = self.cfg.get('dropout', 0.1)
    assert len(block_widths) == len(block_strides), "block_widths and block_strides must match"

    # stem: standard 3x3 conv (cheap at 1 input channel)
    layers = [
      nn.Conv2d(self.cfg['input_shape'][0], stem_ch, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(stem_ch),
      nn.ReLU(),
    ]

    # depthwise-separable blocks
    in_ch = stem_ch
    for out_ch, stride in zip(block_widths, block_strides):
      layers.append(DepthwiseSeparableBlock(in_ch, out_ch, stride=stride))
      in_ch = out_ch

    # global average pooling
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    self.features = nn.Sequential(*layers)

    # classifier head
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(dropout),
      nn.Linear(in_ch, head_dim),
      nn.ReLU(),
      nn.Linear(head_dim, self.cfg['num_classes']),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


class SlimCNNFiLM(ModelBase):
  """
  Track C2 student: SlimCNN conditioned on MSAB modulation features via FiLM
  (the BioME idea ported to a tiny CNN). The MSAB context vector is BatchNorm-
  standardized (the raw values are tiny) and injected after each depthwise-
  separable block through a per-block FiLM2d layer. forward takes (mel, ctx).
  """

  def define_network_structure(self):

    assert len(self.cfg['input_shape']) == 3
    self.needs_context = True   # signals the training loop to feed the MSAB ctx

    stem_ch = self.cfg.get('stem_ch', 24)
    block_widths = self.cfg.get('block_widths', [48, 64, 96, 128])
    block_strides = self.cfg.get('block_strides', [2, 2, 2, 1])
    head_dim = self.cfg.get('head_dim', 64)
    dropout = self.cfg.get('dropout', 0.1)
    self.ctx_dim = self.cfg.get('ctx_dim', 258)
    ctx_proj_dim = self.cfg.get('ctx_proj_dim', 32)
    assert len(block_widths) == len(block_strides), "block_widths and block_strides must match"

    # stem
    self.stem = nn.Sequential(
      nn.Conv2d(self.cfg['input_shape'][0], stem_ch, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(stem_ch),
      nn.ReLU(),
    )

    # context path: standardize the (tiny-valued) MSAB, then project to a
    # compact shared context so the per-block FiLM modulators stay cheap
    self.ctx_norm = nn.BatchNorm1d(self.ctx_dim)
    self.ctx_proj = nn.Sequential(nn.Linear(self.ctx_dim, ctx_proj_dim), nn.ReLU())

    # DS blocks, each followed by a FiLM2d conditioned on the projected context
    self.blocks = nn.ModuleList()
    self.films = nn.ModuleList()
    in_ch = stem_ch
    for out_ch, stride in zip(block_widths, block_strides):
      self.blocks.append(DepthwiseSeparableBlock(in_ch, out_ch, stride=stride))
      self.films.append(FiLM2d(out_ch, ctx_proj_dim))
      in_ch = out_ch

    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Dropout(dropout),
      nn.Linear(in_ch, head_dim),
      nn.ReLU(),
      nn.Linear(head_dim, self.cfg['num_classes']),
    )

  def forward(self, x, ctx=None):
    assert ctx is not None, "SlimCNNFiLM requires the MSAB context vector"
    ctx = self.ctx_proj(self.ctx_norm(ctx))
    x = self.stem(x)
    for block, film in zip(self.blocks, self.films):
      x = film(block(x), ctx)
    x = self.pool(x)
    return self.classifier(x)

  # -- two-input variants of the base counting / export helpers --------------

  def _torchinfo_total(self, col):
    import torchinfo
    was_training = self.training
    self.eval()
    dummy = (torch.zeros((1,) + tuple(self.cfg['input_shape']), device=self.device),
             torch.zeros(1, self.ctx_dim, device=self.device))
    total = getattr(torchinfo.summary(self, input_data=dummy, col_names=[col], verbose=0),
                    'total_params' if col == 'num_params' else 'total_mult_adds')
    if was_training: self.train()
    return total

  def count_params(self): return self._torchinfo_total('num_params')
  def count_operations(self): return self._torchinfo_total('mult_adds')

  def save_model_to_tflite(self):
    # C2 is research-first: the deployable path needs a 2-input tflite plus an
    # on-device MSAB kernel. Defer until the host-side gain justifies it.
    print("\n*** SlimCNNFiLM: tflite export deferred (research phase). "
          "float .pth metrics are logged; build the 2-input/on-device path only if MSAB-FiLM wins.")
    return


class EffNetB3Slim(ModelBase):
  """
  Perch-shaped student: a slimmed EfficientNet-B3 — the backbone Perch v2 uses
  (chirp/models/perch_2.py) — narrowed to ~200k params, on the native PCEN
  input. Keeps B3's depth/block structure (depth_multiplier=1.4) and only slims
  the *width* (channel_multiplier=0.13 -> ~190k backbone params). in_chans=1
  takes the (1,40,133) PCEN spectrogram directly (EfficientNet's 32x downsample
  -> ~2x5 map -> global-pooled embedding), so no resize is needed. Trained from
  scratch (no ImageNet weights exist at this width) with the same Perch
  distillation as the best CNN, to test whether a teacher-shaped student helps.
  """

  def define_network_structure(self):
    from timm.models.efficientnet import _gen_efficientnet

    assert len(self.cfg['input_shape']) == 3
    cm = self.cfg.get('channel_multiplier', 0.13)
    dm = self.cfg.get('depth_multiplier', 1.4)     # keep B3 depth; slim width only
    head_dim = self.cfg.get('head_dim', 64)
    dropout = self.cfg.get('dropout', 0.2)

    # slimmed B3 backbone -> pooled mean embedding (num_classes=0)
    self.backbone = _gen_efficientnet(
      'efficientnet_b3', channel_multiplier=cm, depth_multiplier=dm,
      in_chans=self.cfg['input_shape'][0], num_classes=0)
    emb_dim = self.backbone.num_features

    self.classifier = nn.Sequential(
      nn.LayerNorm(emb_dim),
      nn.Dropout(dropout),
      nn.Linear(emb_dim, head_dim),
      nn.ReLU(),
      nn.Linear(head_dim, self.cfg['num_classes']),
    )

  def forward(self, x):
    return self.classifier(self.backbone(x))   # x: (B,1,C,T) -> emb -> logits

  def forward_with_spatial(self, x):
    """
    Layer-to-layer distillation hook: returns (logits, spatial_map). The spatial
    map is the pre-pool B3 feature map (B, num_features, H, W) -- the student
    analogue of Perch's `spatial_embedding`, matched in spatial size by the
    Perch-resolution front-end so a hint loss can compare them (channels are
    projected). Same forward path as forward(), just also exposing the map.
    """
    feat = self.backbone.forward_features(x)       # (B, num_features, H, W)
    emb = self.backbone.forward_head(feat)         # global-pool -> (B, num_features)
    return self.classifier(emb), feat

  # torchinfo can undercount timm blocks; use the exact parameter sum
  def count_params(self):
    return sum(p.numel() for p in self.parameters())

  def save_model_to_tflite(self):
    # research-first: EfficientNet (SiLU + squeeze-excite) may not lower cleanly
    # to int8/esp-nn. Don't let an export failure crash a training run.
    try:
      return super().save_model_to_tflite()
    except Exception as e:
      print("\n*** EffNetB3Slim: tflite export deferred ({}). "
            "float .pth metrics are logged.".format(type(e).__name__))
      return



if __name__ == '__main__':
  """
  model tiny ml
  """

  import yaml

  # yaml config file
  cfg = yaml.safe_load(open(Path(__file__).parent.parent / 'config.yaml'))

  # params
  num_classes = 11
  num_samples = 4

  # test data sample
  y = torch.randint(0, num_classes-1, (num_samples,))

  # model
  x = torch.randn(num_samples, 1, 133, 40)
  model = Baseline(cfg['pytorch_framework']['model'], input_shape=tuple(x.shape[1:]), num_classes=num_classes)
  model.info()

  # data structure
  data = (x, y)

  # to train mode
  model.set_model_to_training_mode()

  # train loop
  for epoch in range(50):

    # train model
    loss = model.train_step(data)
    print("Epoch {:03} with loss: {:6f}".format(epoch + 1, loss))

  # eval mode (must be done to disable for instance dropout)
  model.set_model_to_evaluation_mode()

  # validation step
  y_pred, loss = model.validation_step(data)

  print("actual: ", y.numpy())
  print("prediction: ", y_pred)
  print("loss: ", loss)
  print("acc: ", np.mean(y.numpy() == np.argmax(y_pred, axis=-1)))

  # save model
  model.save()
