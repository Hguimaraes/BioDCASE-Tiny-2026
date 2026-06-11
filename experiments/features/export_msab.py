# --
# MSAB (modulation spectrogram average bands) export
#
# computes the MSAB context vector for every clip and caches it keyed by wav
# stem, so it can be attached to the student (FiLM side-channel) the same way
# the Perch teacher logits are. Pure torch (no Perch); runs in the main .venv.
#
#   .venv/bin/python experiments/features/export_msab.py \
#       --data-root <dataset> --split Train Validation
#
# Output per split: experiments/features/msab/<tag>/<split>.npz with
#   stems (N,), msab (N, msab_dim), labels (N,)

import sys
import argparse
import numpy as np
import soundfile as sf
import torch

from pathlib import Path

if __name__ == '__main__': [sys.path.append(p) for p in [str(Path(__file__).parent.parent.parent)] if p not in sys.path]
from biodcase_tiny.feature_extraction.modulation_spectrum import ModulationSpectrum


def parse_args():
  p = argparse.ArgumentParser(description='Export MSAB features for a dataset.')
  p.add_argument('--data-root', required=True)
  p.add_argument('--split', nargs='+', default=['Train', 'Validation'])
  p.add_argument('--out', default='experiments/features/msab')
  p.add_argument('--file-ext', default='.wav')
  p.add_argument('--sample-rate', type=int, default=24000)
  p.add_argument('--duration-s', type=float, default=3.0)
  p.add_argument('--n-fft1', type=int, default=256)
  p.add_argument('--win-size', type=int, default=256)
  p.add_argument('--win-shift', type=int, default=128)
  p.add_argument('--n-fft2', type=int, default=256)
  p.add_argument('--tag', default='mss_nfft256')
  p.add_argument('--limit', type=int, default=0)
  return p.parse_args()


def discover_clips(split_dir, file_ext):
  clips = []
  for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
    for wav in sorted(class_dir.glob('*{}'.format(file_ext))):
      clips.append((class_dir.name, wav))
  return clips


def load_clip(path, sr, n_samples):
  x, fs = sf.read(path, dtype='float32', always_2d=False)
  if x.ndim > 1: x = x.mean(axis=1)
  assert fs == sr, 'expected {} Hz, got {} for {}'.format(sr, fs, path)
  if len(x) < n_samples: x = np.pad(x, (0, n_samples - len(x)))
  elif len(x) > n_samples: x = x[:n_samples]
  return x


def main():
  args = parse_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ms = ModulationSpectrum(sample_rate=args.sample_rate, n_fft1=args.n_fft1,
                          win_size=args.win_size, win_shift=args.win_shift, n_fft2=args.n_fft2).to(device)
  print('MSAB extractor on {} | msab_dim={}'.format(device, ms.msab_dim))

  data_root = Path(args.data_root)
  out_dir = Path(args.out) / args.tag
  out_dir.mkdir(parents=True, exist_ok=True)
  n_samples = int(args.duration_s * args.sample_rate)

  # consistent label dict across splits
  all_labels, split_clips = set(), {}
  for split in args.split:
    clips = discover_clips(data_root / split, args.file_ext)
    if args.limit: clips = clips[:args.limit]
    split_clips[split] = clips
    all_labels.update(name for name, _ in clips)
  label_dict = {name: i for i, name in enumerate(sorted(all_labels))}

  for split, clips in split_clips.items():
    print('\nExporting MSAB for [{}] - {} clips'.format(split, len(clips)))
    stems, feats, labels = [], [], []
    batch_wavs, batch_meta = [], []

    def flush():
      if not batch_wavs: return
      w = torch.from_numpy(np.stack(batch_wavs)).to(device)
      m = ms(w).cpu().numpy().astype(np.float32)
      for i, (stem, lab) in enumerate(batch_meta):
        stems.append(stem); feats.append(m[i]); labels.append(lab)
      batch_wavs.clear(); batch_meta.clear()

    for i, (label_name, wav_path) in enumerate(clips):
      batch_wavs.append(load_clip(wav_path, args.sample_rate, n_samples))
      batch_meta.append((wav_path.stem, label_dict[label_name]))
      if len(batch_wavs) >= 64: flush()
      if (i + 1) % 200 == 0 or (i + 1) == len(clips): print('  {}/{}'.format(i + 1, len(clips)))
    flush()

    out_path = out_dir / '{}.npz'.format(split)
    np.savez_compressed(out_path, stems=np.array(stems), msab=np.stack(feats).astype(np.float32), labels=np.array(labels, dtype=np.int64))
    print('  saved -> {} | msab {}'.format(out_path, np.stack(feats).shape))

  print('\nDone. MSAB written under: {}'.format(out_dir))


if __name__ == '__main__':
  main()
