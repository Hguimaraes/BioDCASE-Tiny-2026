# --
# Slice downloaded focal recordings into fixed 3 s clips for teacher labeling.
#
# Turns the variable-length mp3s under <in-dir>/<class>/ into uniform 3 s, 24 kHz
# mono wav clips under <out-dir>/<class>/, ready for the Perch export + feature
# pipeline (same class-subfolder layout as the dataset). Steps per recording:
#   1. decode mp3 + resample to 24 kHz mono (soundfile/librosa, no ffmpeg needed),
#   2. window into clip_sec frames at hop_sec (hop < clip_sec => overlap, which
#      lets scarce classes like Mallard reach the per-class target),
#   3. energy VAD: drop near-silent windows (focal recordings have long gaps),
#   4. write up to --max-clips-per-rec per recording, --target-per-class overall,
#      shuffling recordings first so clips are diverse across individuals/sites.
#
# Domain note: these are FOCAL clips (single bird, close mic); they will be
# Perch-labeled, not trusted by their species folder. Output is balanced-ish by
# the per-class cap so no single common species dominates the extra set.
#
#   python experiments/data/slice_audio.py \
#       --in-dir /home/hguimaraes/datasets/extra/xc/raw \
#       --out-dir /home/hguimaraes/datasets/extra/xc/clips \
#       --target-per-class 6000 --hop-sec 1.5

import sys
import re
import math
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from experiments.data.species import LABELS

AUDIO_EXTS = ('.mp3', '.wav', '.flac', '.ogg')


def parse_args():
  p = argparse.ArgumentParser(description='Slice focal recordings into fixed 3 s clips.')
  p.add_argument('--in-dir', required=True, help='root of <class>/<recording> audio')
  p.add_argument('--out-dir', required=True, help='root for <class>/<clip>.wav output')
  p.add_argument('--sr', type=int, default=24000, help='target sample rate (dataset = 24000)')
  p.add_argument('--clip-sec', type=float, default=3.0, help='clip length in seconds')
  p.add_argument('--hop-sec', type=float, default=2.5, help='window hop; < clip-sec => overlap (2.5 = 0.5 s overlap)')
  p.add_argument('--target-per-class', type=int, default=0, help='clip budget per class spread evenly over ALL recordings (0 = take ALL windows from every recording, let Perch filter later)')
  p.add_argument('--clips-per-rec', type=int, default=0, help='clips per recording (0 = auto from target, or ALL when target=0)')
  p.add_argument('--max-clips-per-rec', type=int, default=0, help='hard cap on per-recording clips (0 = no cap)')
  # light VAD: drop windows that are MOSTLY silence; not aggressive energy gating.
  p.add_argument('--vad-min-active', type=float, default=0.1, help='keep a window if >= this fraction of its 100 ms frames have sound')
  p.add_argument('--vad-floor-factor', type=float, default=0.15, help='sound floor as a fraction of the recording p90 frame level (adapts to quiet recordings)')
  p.add_argument('--silence-floor', type=float, default=1e-3, help='absolute floor (drop dead silence)')
  p.add_argument('--no-vad', action='store_true', help='disable VAD (keep all windows)')
  p.add_argument('--species', nargs='+', default=None, help='subset of class folders (default: all present)')
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args()


def load_audio(path, sr):
  import librosa
  y, _ = librosa.load(str(path), sr=sr, mono=True)
  return y.astype(np.float32)


def iter_windows(y, w, h):
  """yield (start_sample, window) of length w; a too-short clip is zero-padded once."""
  if len(y) < w:
    yield 0, np.pad(y, (0, w - len(y)))
    return
  for start in range(0, len(y) - w + 1, h):
    yield start, y[start:start + w]


def keep_windows(y, w, h, sr, use_vad, min_active, floor_factor, abs_floor, frame_sec=0.1):
  """window a recording; LIGHT VAD drops windows that are mostly silence.

  A window is kept if at least `min_active` of its 100 ms frames exceed a sound
  floor set relative to the recording's own loud level (p90 frame RMS), so quiet
  recordings still pass and we only cut near-silent / tiny-sound-region windows.
  Deliberately lenient -- the Perch agreement filter does the real selection.
  """
  wins = list(iter_windows(y, w, h))
  if not wins or not use_vad:
    return wins
  fl = max(1, int(sr * frame_sec))
  n = len(y) // fl
  if n == 0:
    return wins
  ref = float(np.percentile(np.sqrt((y[:n * fl].reshape(n, fl) ** 2).mean(1) + 1e-12), 90))
  floor = max(abs_floor, floor_factor * ref)
  out = []
  for start, win in wins:
    m = len(win) // fl
    if m == 0:
      out.append((start, win)); continue
    frms = np.sqrt((win[:m * fl].reshape(m, fl) ** 2).mean(1) + 1e-12)
    if float((frms > floor).mean()) >= min_active:
      out.append((start, win))
  return out


def evenly_spaced(items, k):
  """pick k items spread across the whole list (not the first k) for within-recording diversity."""
  if k >= len(items):
    return items
  idx = sorted(set(int(round(i)) for i in np.linspace(0, len(items) - 1, num=k)))
  return [items[i] for i in idx]


def class_dirs(in_dir, species):
  present = [d for d in sorted(Path(in_dir).iterdir()) if d.is_dir()]
  if species:
    want = set(species); present = [d for d in present if d.name in want]
  return present


def main():
  args = parse_args()
  random.seed(args.seed)
  w, h = int(args.clip_sec * args.sr), max(1, int(args.hop_sec * args.sr))
  in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
  print('Slicing {} -> {} | {:.0f}s clips @ {} Hz | hop {:.1f}s | VAD {} | target/class {}'.format(
      in_dir, out_dir, args.clip_sec, args.sr, args.hop_sec, 'off' if args.no_vad else 'on',
      args.target_per_class or 'unbounded'))

  summary = []
  for cdir in class_dirs(in_dir, args.species):
    label = cdir.name
    files = sorted(f for f in cdir.rglob('*') if f.suffix.lower() in AUDIO_EXTS)  # SORT before seeded shuffle
    random.shuffle(files)
    odir = out_dir / label; odir.mkdir(parents=True, exist_ok=True)
    # clips/recording: None => take ALL windows from every recording (diversity;
    # Perch filters later). Otherwise spread the class budget evenly over ALL recs.
    if args.clips_per_rec > 0:
      per_rec = args.clips_per_rec
    elif args.target_per_class > 0:
      per_rec = max(1, math.ceil(args.target_per_class / max(1, len(files))))
    else:
      per_rec = None
    if per_rec is not None and args.max_clips_per_rec > 0:
      per_rec = min(per_rec, args.max_clips_per_rec)
    written, used, errors = 0, 0, 0
    for f in files:                                 # process ALL recordings, no early stop
      try:
        y = load_audio(f, args.sr)
      except Exception:
        errors += 1; continue
      kept = keep_windows(y, w, h, args.sr, not args.no_vad,
                          args.vad_min_active, args.vad_floor_factor, args.silence_floor)
      if not kept:
        continue
      used += 1
      # clean clip id: the XC number if present, else an ascii-slugged stem
      m = re.match(r'(XC\d+)', f.stem)
      cid = m.group(1) if m else re.sub(r'[^A-Za-z0-9]+', '_', f.stem)[:40]
      sel = kept if per_rec is None else evenly_spaced(kept, per_rec)  # spread within the recording
      for start, win in sel:
        # deterministic name: <XCid>_<start_ms>.wav (idempotent re-runs)
        out_path = odir / '{}_{:07d}.wav'.format(cid, int(start * 1000 / args.sr))
        sf.write(out_path, win, args.sr, subtype='PCM_16')
        written += 1
    print('  {:<26} {:>6} clips from {:>4}/{} recordings ({}/rec){}'.format(
        label, written, used, len(files), 'all' if per_rec is None else per_rec,
        ' | {} decode errors'.format(errors) if errors else ''))
    summary.append((label, written))

  total = sum(n for _, n in summary)
  print('\nDone: {} clips across {} classes under {}'.format(total, len(summary), out_dir))
  # clips drawn from every recording (diversity); count lands near target. classes
  # well under target -> recordings too short/silent: raise --clips-per-rec or lower --hop-sec.
  short = [l for l, n in summary if args.target_per_class and n < 0.9 * args.target_per_class]
  if short:
    print('Under target ({}): {} -> raise --clips-per-rec or lower --hop-sec'.format(
        args.target_per_class, ', '.join(short)))


if __name__ == '__main__':
  main()
