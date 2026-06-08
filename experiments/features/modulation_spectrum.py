"""BioME-inspired 2D modulation-spectrum feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ModulationSpectrum2DFeatureHandler:
  """Extract a full acoustic-frequency x modulation-frequency PSD map.

  This follows the BioME modulation-spectrum path, but deliberately keeps the
  2D map instead of averaging over the acoustic and modulation axes.
  """

  def __init__(self, cfg: dict | None = None, **kwargs):
    self.cfg = {
      "target_sample_rate": 24000,
      "mss_n_fft1": 1024,
      "mss_n_fft2": None,
      "mss_win_size": 600,
      "mss_win_shift": 240,
      "log_scale": True,
      "normalize_features": True,
      "normalize_method": "minmax",
      "percentile_clip": None,
      "modulation_dc_mode": "keep",
      "eps": 1e-8,
      "resize_shape": None,
      "add_channel_dimension": True,
    }
    self.cfg.update(cfg or {})
    self.cfg.update(kwargs)

  def extract(self, x):
    wav = self._to_waveform_tensor(x)
    features = self._modulation_spectrum(wav)[0]

    if self.cfg["log_scale"]:
      features = torch.log1p(features)

    features = self._process_modulation_dc(features)
    features = self._clip_percentiles(features)

    resize_shape = self.cfg.get("resize_shape")
    if resize_shape is not None:
      features = F.interpolate(
        features.unsqueeze(0).unsqueeze(0),
        size=tuple(resize_shape),
        mode="bilinear",
        align_corners=False,
      ).squeeze(0).squeeze(0)

    if self.cfg["normalize_features"]:
      features = self._normalize_features(features)

    out = features.cpu().numpy().astype(np.float32)
    if self.cfg["add_channel_dimension"]:
      out = out[np.newaxis, :]
    return out

  def _process_modulation_dc(self, features: torch.Tensor) -> torch.Tensor:
    mode = self.cfg.get("modulation_dc_mode", "keep")
    if mode == "keep":
      return features
    if mode == "drop":
      return features[:, 1:]
    if mode == "zero":
      features = features.clone()
      features[:, 0] = 0.0
      return features
    raise ValueError(f"Unsupported modulation_dc_mode: {mode}")

  def _clip_percentiles(self, features: torch.Tensor) -> torch.Tensor:
    percentile_clip = self.cfg.get("percentile_clip")
    if percentile_clip is None:
      return features
    low, high = percentile_clip
    if low is None and high is None:
      return features
    flat = features.flatten()
    min_value = torch.quantile(flat, float(low) / 100.0) if low is not None else features.amin()
    max_value = torch.quantile(flat, float(high) / 100.0) if high is not None else features.amax()
    return torch.clamp(features, min=min_value, max=max_value)

  def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
    method = self.cfg.get("normalize_method", "minmax")
    eps = self.cfg["eps"]
    if method == "none":
      return features
    if method == "minmax":
      x_min = features.amin()
      x_range = features.amax() - x_min
      return (features - x_min) / torch.clamp(x_range, min=eps)
    if method == "zscore":
      return (features - features.mean()) / torch.clamp(features.std(), min=eps)
    if method == "robust_zscore":
      median = features.median()
      mad = (features - median).abs().median()
      return (features - median) / torch.clamp(1.4826 * mad, min=eps)
    raise ValueError(f"Unsupported normalize_method: {method}")

  def _to_waveform_tensor(self, x) -> torch.Tensor:
    if not isinstance(x, np.ndarray):
      x = np.asarray(x)
    if x.ndim > 1:
      x = np.mean(x, axis=-1)
    x = x.astype(np.float32, copy=False)
    if np.issubdtype(x.dtype, np.integer):
      x = x / np.iinfo(x.dtype).max
    elif np.max(np.abs(x)) > 1.5:
      x = x / np.iinfo(np.int16).max
    return torch.from_numpy(x).unsqueeze(0)

  def _normalize_fft(
    self,
    spec_data: torch.Tensor,
    window: torch.Tensor,
    n_samples: int,
    n_fft: int,
    fs: float,
  ) -> tuple[float, torch.Tensor]:
    win_rms = torch.sqrt(window.pow(2.0).sum() / n_samples)
    spec_data = spec_data / win_rms
    spec_data = spec_data.abs().pow(2.0)
    spec_data = spec_data * (1.0 / n_fft**2)

    if n_fft % 2 != 0:
      spec_data[:, 1:, :] *= 2
    else:
      spec_data[:, 1:-1, :] *= 2

    f_delta = fs / n_fft
    spec_data = spec_data / f_delta
    return f_delta, spec_data

  @torch.no_grad()
  def _modulation_spectrum(self, wavs: torch.Tensor) -> torch.Tensor:
    _, n_samples = wavs.shape
    device = wavs.device

    window = torch.hamming_window(
      self.cfg["mss_win_size"],
      periodic=True,
      device=device,
    )
    spec_data = torch.stft(
      wavs,
      n_fft=self.cfg["mss_n_fft1"],
      win_length=self.cfg["mss_win_size"],
      hop_length=self.cfg["mss_win_shift"],
      window=window,
      return_complex=True,
      onesided=True,
    )
    _, _, n_windows = spec_data.shape

    _, spec_data = self._normalize_fft(
      spec_data,
      window,
      n_samples,
      self.cfg["mss_n_fft1"],
      self.cfg["target_sample_rate"],
    )

    fs_mod = 1 / (self.cfg["mss_win_shift"] / self.cfg["target_sample_rate"])
    n_fft2 = self.cfg["mss_n_fft2"] or n_windows
    window = torch.hamming_window(n_windows, periodic=True, device=device)
    spec_data = torch.sqrt(torch.clamp(spec_data, min=0.0)) * window
    mod_psd = torch.fft.rfft(spec_data, n=n_fft2, dim=2)

    _, mod_psd = self._normalize_fft(
      mod_psd,
      window,
      n_samples,
      n_fft2,
      fs_mod,
    )
    return mod_psd
