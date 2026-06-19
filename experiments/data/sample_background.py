# --
# Sample N background clips from a large audio pool (e.g. TAU Urban Acoustic
# Scenes) -- one fixed-length window per file, no per-file segmentation.
#
# Produces the Background class for the 11-class task: 3 s / 24 kHz / mono wav,
# in a class-subfolder layout so it joins the bird clips for Perch labeling /
# distillation. TAU is passive urban soundscape => domain-matched "no bird"
# background. Random per-file offset gives variety; deterministic by --seed.
#
#   .venv/bin/python experiments/data/sample_background.py \
#       --in-dir  /media/hguimaraes/Expansion/datasets/TAU-urban-acoustic-scenes-2022-mobile/TAU-urban-acoustic-scenes-2022-mobile-development/audio \
#       --out-dir /home/hguimaraes/datasets/extra/xc/clips/Background \
#       --n 10000

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

try:
  from tqdm import tqdm
except ImportError:
  def tqdm(it, **k): return it


def load_audio(path, sr):
  import librosa
  y, _ = librosa.load(str(path), sr=sr, mono=True)
  return y.astype(np.float32)


def main():
  ap = argparse.ArgumentParser(description='Sample N background clips, one window per file.')
  ap.add_argument('--in-dir', required=True, help='root of source audio (searched recursively)')
  ap.add_argument('--out-dir', required=True, help='destination (the Background class folder)')
  ap.add_argument('--n', type=int, default=10000, help='target number of clips')
  ap.add_argument('--sr', type=int, default=24000)
  ap.add_argument('--clip-sec', type=float, default=3.0)
  ap.add_argument('--offset', choices=['random', 'center', 'start'], default='random', help='where to take the window')
  ap.add_argument('--ext', default='.wav', help='source audio extension')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()

  random.seed(args.seed); rng = np.random.default_rng(args.seed)
  w = int(args.clip_sec * args.sr)
  files = sorted(Path(args.in_dir).rglob('*' + args.ext))
  if not files:
    print('no {} files under {}'.format(args.ext, args.in_dir)); return
  random.shuffle(files)                                  # deterministic given --seed
  out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
  existing = {p.stem for p in out.glob('*.wav')}         # resume: count what's already there
  print('found {} source files | target {} | already present {}'.format(len(files), args.n, len(existing)))

  written, errors = 0, 0
  for f in tqdm(files, unit='file'):
    if len(existing) + written >= args.n:
      break
    if f.stem in existing:
      continue
    try:
      y = load_audio(f, args.sr)
    except Exception:
      errors += 1; continue
    if len(y) < w:
      y = np.pad(y, (0, w - len(y))); start = 0          # short file -> pad
    elif args.offset == 'random':
      start = int(rng.integers(0, len(y) - w + 1))
    elif args.offset == 'center':
      start = (len(y) - w) // 2
    else:
      start = 0
    sf.write(out / (f.stem + '.wav'), y[start:start + w], args.sr, subtype='PCM_16')
    written += 1

  total = len(existing) + written
  print('wrote {} new clips -> {} | total now {} ({} errors)'.format(written, out, total, errors))
  if total < args.n:
    print('NOTE: only {} available (< target {}).'.format(total, args.n))


if __name__ == '__main__':
  main()
