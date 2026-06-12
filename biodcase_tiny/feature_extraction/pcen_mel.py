# --
# PCEN-mel front-end (Track D: match Perch's input compression).
#
# Perch's frontend is a mel spectrogram with PCEN (Per-Channel Energy
# Normalization) compression, NOT the int log-mel our device pipeline uses
# (see chirp/models/frontend.py PCENScalingConfig and chirp/audio_utils.py:pcen
# in the user's perch repo). PCEN is a per-channel AGC: an EMA-smoothed energy
# estimate normalizes each band, suppressing stationary noise and compressing
# dynamic range. This module ports it to torch (CPU/GPU, batchable) so the tiny
# student can be trained on a Perch-like front-end and compared against the
# int log-mel at the SAME framing (window/stride/mel) — a clean log-vs-PCEN A/B.
#
# We deliberately keep OUR framing (24 kHz, 40 mel, win 4096 / hop 512 -> 133
# frames) rather than Perch's 32 kHz/128-mel/100 fps: matching Perch's *mel
# resolution* would ~7x the input tensor (breaking the baseline resource
# envelope) and would not help logit-level distillation anyway (the teacher
# logits are fixed, computed by Perch's own internal frontend on raw audio).
# What is meaningful and free is matching the PCEN *compression* and its
# bioacoustic params (Lostanlen "PCEN: Why and How"), which we do below.

import numpy as np
import torch
import torchaudio


# Perch bioacoustic PCEN params (chirp presets.py:get_bio_pcen_melspec_config)
PCEN_DEFAULTS = dict(smoothing_coef=0.145, gain=0.8, bias=10.0, root=4.0, eps=1e-6)


def pcen(energy: torch.Tensor, smoothing_coef=0.145, gain=0.8, bias=10.0,
         root=4.0, eps=1e-6) -> torch.Tensor:
  """
  Per-Channel Energy Normalization. Ported from chirp/audio_utils.py:pcen.

  energy: (..., n_mels, n_frames) magnitude/energy mel spectrogram. The EMA
  smoother runs causally along the frame (time) axis, per mel channel.
  Returns the PCEN-compressed spectrogram, same shape.
  """
  # causal EMA along time (last axis), initial state = first frame
  s = smoothing_coef
  m = torch.empty_like(energy)
  m[..., 0] = energy[..., 0]
  for t in range(1, energy.shape[-1]):
    m[..., t] = (1.0 - s) * m[..., t - 1] + s * energy[..., t]

  inv_root = 1.0 / root
  out = (energy / (eps + m) ** gain + bias) ** inv_root - bias ** inv_root
  return out


class PCENMel:
  """
  PCEN-mel feature extractor matching the int log-mel framing.

  Produces a (n_mels, n_frames) float feature for a 1-D waveform, using the
  same window/stride/mel-count as the device log-mel path so the only variable
  vs the baseline is log -> PCEN. freq range follows Perch (50 Hz .. Nyquist),
  power=1.0 (magnitude mel) as in Perch's SimpleMelspec.
  """

  def __init__(self, sample_rate=24000, window_len=4096, window_stride=512,
               n_mels=40, f_min=50.0, f_max=None, power=1.0, n_fft=None,
               resample_to=None, pad_seconds=None, pcen_kwargs=None, device='cpu'):
    self.sample_rate = sample_rate
    self.device = device
    self.pcen_kwargs = {**PCEN_DEFAULTS, **(pcen_kwargs or {})}

    # optional resample (e.g. 24k -> Perch's 32k) and pad/trim to a fixed window
    # (e.g. 5 s) so the spectrogram matches the teacher's frontend resolution.
    self.resampler = (torchaudio.transforms.Resample(sample_rate, resample_to).to(device)
                      if resample_to and resample_to != sample_rate else None)
    eff_sr = resample_to or sample_rate
    self.pad_samples = int(round(pad_seconds * eff_sr)) if pad_seconds else None
    n_fft = n_fft or window_len
    f_max = f_max if f_max is not None else eff_sr / 2

    # mel spectrogram. center=False + unpadded sliding window; for the device
    # path (40 mel) this matches process_window framing (133 frames). For the
    # Perch-resolution path (128 mel, 32k, hop 320) it yields ~500 frames.
    self.melspec = torchaudio.transforms.MelSpectrogram(
      sample_rate=eff_sr, n_fft=n_fft, win_length=window_len,
      hop_length=window_stride, f_min=f_min, f_max=f_max, n_mels=n_mels,
      power=power, center=False, norm='slaney', mel_scale='slaney',
    ).to(device)

  def __call__(self, wav: torch.Tensor) -> torch.Tensor:
    # wav: (T,) or (B, T) float in [-1, 1]
    single = wav.ndim == 1
    if single:
      wav = wav.unsqueeze(0)
    if self.resampler is not None:
      wav = self.resampler(wav)
    if self.pad_samples is not None:                       # pad/trim to fixed window
      T = wav.shape[-1]
      if T < self.pad_samples:
        wav = torch.nn.functional.pad(wav, (0, self.pad_samples - T))
      elif T > self.pad_samples:
        wav = wav[..., :self.pad_samples]
    mel = self.melspec(wav)                     # (B, n_mels, n_frames)
    out = pcen(mel, **self.pcen_kwargs)         # (B, n_mels, n_frames)
    return out.squeeze(0) if single else out

  def extract_numpy(self, wav_int16: np.ndarray) -> np.ndarray:
    """
    Convenience for the FeatureHandler: int16/float ndarray -> (n_mels, frames)
    float32 ndarray (pre-normalization; the handler does the [0,1] min-max).
    """
    x = np.asarray(wav_int16)
    if np.issubdtype(x.dtype, np.integer):
      x = x.astype(np.float32) / np.iinfo(np.int16).max
    else:
      x = x.astype(np.float32)
      peak = np.max(np.abs(x)) or 1.0
      x = x / peak
    with torch.no_grad():
      feat = self(torch.from_numpy(x).to(self.device))
    return feat.cpu().numpy().astype(np.float32)


if __name__ == '__main__':
  # quick shape check against the expected (40, 133)
  pm = PCENMel()
  wav = (np.random.randn(72000) * 3000).astype(np.int16)
  f = pm.extract_numpy(wav)
  print('PCEN-mel feature shape:', f.shape, '| dtype', f.dtype,
        '| range [{:.3f}, {:.3f}]'.format(f.min(), f.max()))
