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


class GaborTemporalConv(nn.Module):
  """Depthwise 1-D modulation filterbank over time: each of `channels` feature
  channels is filtered by `n_mod` Gabor band-passes initialized at modulation
  rates f_min..f_max Hz -> channels*n_mod modulation maps. It's a grouped Conv1d
  (Gabor-initialized but learnable; freeze with learnable=False), so it exports to
  tflite -- the deployable stand-in for an FFT along the feature-map time axis."""
  def __init__(self, channels, n_mod=8, kernel=33, fs_mod=46.875, f_min=1.0, f_max=20.0, learnable=True):
    super().__init__()
    self.conv = nn.Conv1d(channels, channels * n_mod, kernel, padding=kernel // 2, groups=channels, bias=False)
    t = torch.arange(kernel).float() - kernel // 2
    sigma = kernel / 6.0
    win = torch.exp(-t ** 2 / (2 * sigma ** 2))
    bank = torch.stack([win * torch.cos(2 * math.pi * (f / fs_mod) * t)        # (n_mod, kernel)
                        for f in torch.linspace(f_min, f_max, n_mod)])
    bank = bank - bank.mean(dim=1, keepdim=True)                               # zero-DC band-pass
    bank = bank / (bank.norm(dim=1, keepdim=True) + 1e-8)
    with torch.no_grad():
      self.conv.weight.copy_(bank.repeat(channels, 1).unsqueeze(1))            # (channels*n_mod, 1, kernel)
    self.conv.weight.requires_grad_(learnable)

  def forward(self, x):                                                        # (B, C, T) -> (B, C*n_mod, T)
    return self.conv(x)


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


class AttnPool1d(nn.Module):
  """Additive attention pooling over time (replaces global avg-pool):
  a_t = softmax(v . tanh(W h_t)); out = sum_t a_t h_t. Deployable."""
  def __init__(self, dim, attn_dim=64):
    super().__init__()
    self.W = nn.Linear(dim, attn_dim); self.v = nn.Linear(attn_dim, 1)
  def forward(self, h):                                                        # (B, dim, T)
    h = h.transpose(1, 2)                                                      # (B, T, dim)
    a = torch.softmax(self.v(torch.tanh(self.W(h))), dim=1)                    # (B, T, 1)
    return (a * h).sum(dim=1)                                                  # (B, dim)


class ModFilterNet(ModelBase):
  """Signal-processing-inductive-bias student: conv stem that pools FREQUENCY only
  (time preserved) -> learnable Gabor temporal modulation filterbank -> attention
  pooling over time -> classifier. Optional 2-D Gabor STRF front conv (cfg strf).
  All standard conv/linear/softmax (no FFT), so it stays deployable. The attention-
  pooled descriptor is the representation the embedding-distill lever regresses on.
  cfg: n_filters (32), n_mod (8), mod_kernel (33, odd), strf (False),
  learnable_mod (True), attn_dim (64), fs_mod (46.875), dropout (0.05)."""

  def define_network_structure(self):
    nf = self.cfg.get('n_filters', 32); n_mod = self.cfg.get('n_mod', 8)
    learn = self.cfg.get('learnable_mod', True)
    front = GaborSTRFConv(nf, learnable=learn) if self.cfg.get('strf', False) else nn.Conv2d(1, nf, 3, padding=1)
    self.stem = nn.Sequential(                                                 # pool FREQ only, keep time
        front, nn.ReLU(), nn.MaxPool2d((2, 1)),
        nn.Conv2d(nf, nf * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
        nn.Conv2d(nf * 2, nf * 2, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, None)))                                       # (B, 2nf, 1, T)
    ch = nf * 2
    self.modfb = GaborTemporalConv(ch, n_mod=n_mod, kernel=self.cfg.get('mod_kernel', 33),
                                   fs_mod=self.cfg.get('fs_mod', 46.875), learnable=learn)
    self.modbn = nn.BatchNorm1d(ch * n_mod)
    self.gap_dim = ch * n_mod
    self.pool = AttnPool1d(self.gap_dim, attn_dim=self.cfg.get('attn_dim', 64))
    self.classifier = nn.Sequential(nn.Dropout(self.cfg.get('dropout', 0.05)),
                                    nn.Linear(self.gap_dim, 32), nn.ReLU(),
                                    nn.Linear(32, self.cfg['num_classes']))

  def _features(self, x):
    z = self.stem(x).squeeze(2)                                               # (B, ch, T)
    z = torch.relu(self.modbn(self.modfb(z)))                                 # (B, ch*n_mod, T) modulation maps
    return self.pool(z)                                                       # (B, ch*n_mod) attention-pooled

  def forward(self, x):
    return self.classifier(self._features(x))

  def forward_with_features(self, x):
    f = self._features(x)
    return self.classifier(f), f                                             # embed distill regresses f

  def save_model_to_tflite(self):
    try:
      return super().save_model_to_tflite()                                  # designed to be deployable; attempt it
    except Exception as e:
      print("\n*** ModFilterNet: tflite export deferred ({}). float .pth metrics are logged.".format(type(e).__name__))
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
