# --
# modulation spectrum + MSAB features (Modulation Spectrogram Average Bands)
#
# adapted from the BioME encoder (speechprotolab, Guimaraes et al.): compute
# a 2-D modulation spectrum X(f, f_mod) from a waveform, then average it along
# both axes to a compact per-clip context vector (MSAB). Used as side-channel
# information injected into the student via FiLM conditioning (BioDCASE Track C2).
#
# Pipeline:
#   1. STFT  -> spectrogram X(t, f), normalized to an orthonormal power spectrum
#   2. amplitude = sqrt(power), windowed, FFT along time -> modulation spectrum
#   3. MSAB = concat( mean over acoustic freq, mean over modulation freq )

import torch
import torch.nn as nn


class ModulationSpectrum(nn.Module):
  """
  compute MSAB features from a batch of waveforms (B, n_samples) -> (B, msab_dim)

  msab_dim = (n_fft1 // 2 + 1) + (n_fft2 // 2 + 1)
  """

  def __init__(self, sample_rate=24000, n_fft1=256, win_size=256, win_shift=128, n_fft2=256):
    super().__init__()
    self.sample_rate = sample_rate
    self.n_fft1 = n_fft1
    self.win_size = win_size
    self.win_shift = win_shift
    self.n_fft2 = n_fft2

  @property
  def msab_dim(self):
    return (self.n_fft1 // 2 + 1) + (self.n_fft2 // 2 + 1)

  @torch.no_grad()
  def _normalize_fft(self, spec_data, window, n_samples, n_fft, fs):
    """orthonormal power-spectrum normalization (BioME normalize_fft)"""
    win_rms = torch.sqrt(window.pow(2.0).sum() / n_samples)
    spec_data = spec_data / win_rms
    spec_data = spec_data.abs().pow(2.0)
    spec_data = spec_data * (1.0 / n_fft ** 2)
    if n_fft % 2 != 0:
      spec_data[:, 1:, :] = spec_data[:, 1:, :] * 2
    else:
      spec_data[:, 1:-1, :] = spec_data[:, 1:-1, :] * 2
    f_delta = fs / n_fft
    spec_data = spec_data / f_delta
    return f_delta, spec_data

  @torch.no_grad()
  def forward(self, wavs):
    """
    wavs: (B, n_samples) float waveform -> (B, msab_dim) MSAB context vector
    """
    if wavs.dim() == 1: wavs = wavs.unsqueeze(0)
    _, n_samples = wavs.shape

    # STEP 1: STFT spectrogram
    window = torch.hamming_window(self.win_size, periodic=True, device=wavs.device)
    spec_data = torch.stft(
      wavs, n_fft=self.n_fft1, win_length=self.win_size, hop_length=self.win_shift,
      window=window, return_complex=True, onesided=True)
    _, _, n_windows = spec_data.shape
    _, spec_data = self._normalize_fft(spec_data, window, n_samples, self.n_fft1, self.sample_rate)

    # STEP 2: modulation features (FFT along the time/window axis)
    fs_mod = 1.0 / (self.win_shift / self.sample_rate)
    n_fft2 = self.n_fft2 if self.n_fft2 is not None else n_windows
    mod_window = torch.hamming_window(n_windows, periodic=True, device=wavs.device)
    amp = torch.sqrt(torch.clamp(spec_data, min=0.0)) * mod_window
    mod_psd = torch.fft.rfft(amp, n=n_fft2, dim=2)
    _, mod_psd = self._normalize_fft(mod_psd, mod_window, n_samples, n_fft2, fs_mod)

    # MSAB: average bands along acoustic-freq (dim=1) and modulation-freq (dim=2)
    return torch.cat([mod_psd.mean(dim=1), mod_psd.mean(dim=2)], dim=1)


if __name__ == '__main__':
  """ smoke test """
  ms = ModulationSpectrum(sample_rate=24000)
  x = torch.randn(4, 72000)            # 4 clips of 3 s @ 24 kHz
  h = ms(x)
  print('msab_dim:', ms.msab_dim, '| output:', tuple(h.shape), '| finite:', bool(torch.isfinite(h).all()))
  # a tone should give a non-degenerate vector
  t = torch.arange(72000) / 24000.0
  tone = torch.sin(2 * torch.pi * 3000 * t).unsqueeze(0)
  print('tone msab range:', round(float(ms(tone).min()), 4), round(float(ms(tone).max()), 4))
