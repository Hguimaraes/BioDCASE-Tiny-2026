# --
# Build 3 s Background clips from TAU Urban Acoustic Scenes (DCASE 2022 Mobile).
#
# TAU dev files are 1 s segments named scene-city-location-clipid-segidx-device.wav;
# consecutive segidx of the same scene-city-location-clipid-device are temporally
# adjacent pieces of one ~10 s recording. We CONCATENATE 3 consecutive segments
# into a seamless 3 s clip (24 kHz mono), sampling diversely via round-robin over
# unique recordings so the N clips aren't device/segment near-duplicates.
#
# TAU is passive urban soundscape -> domain-matched "no target bird" background.
# (Park scenes may contain birds; Perch-label these later and keep only those it
# calls Background, same agreement idea as the XC birds.)
#
#   .venv/bin/python experiments/data/make_background_from_tau.py \
#       --in-dir  /home/hguimaraes/datasets/extra/tau-2022/TAU-urban-acoustic-scenes-2022-mobile-development/audio \
#       --out-dir /home/hguimaraes/datasets/extra/xc/clips/Background \
#       --n 10000

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

try:
  from tqdm import tqdm
except ImportError:
  def tqdm(it, **k): return it


def parse(stem):
  """scene-city-location-clipid-segidx-device -> (content_key, device, segidx) or None."""
  t = stem.split('-')
  if len(t) < 6:
    return None
  try:
    seg = int(t[4])
  except ValueError:
    return None
  return (t[0], t[1], t[2], t[3]), t[5], seg          # (content, device, segidx)


def main():
  ap = argparse.ArgumentParser(description='Concatenate TAU 1 s segments into 3 s background clips.')
  ap.add_argument('--in-dir', required=True, help='TAU .../audio dir of 1 s segment wavs')
  ap.add_argument('--out-dir', required=True, help='Background class folder')
  ap.add_argument('--n', type=int, default=10000)
  ap.add_argument('--sr', type=int, default=24000)
  ap.add_argument('--n-seg', type=int, default=3, help='segments to concatenate (1 s each -> n_seg seconds)')
  ap.add_argument('--seed', type=int, default=0)
  args = ap.parse_args()
  random.seed(args.seed)

  # index segments by recording (content+device)
  recs = {}                                            # (content, device) -> {segidx: path}
  files = sorted(Path(args.in_dir).glob('*.wav'))      # SORT: glob order is filesystem-dependent
  for f in files:
    p = parse(f.stem)
    if p is None:
      continue
    content, device, seg = p
    recs.setdefault((content, device), {})[seg] = f

  # per-content candidate triples = non-overlapping consecutive runs, across devices
  cands = {}                                           # content -> [(rec_key, [segidx,...]), ...]
  for rec_key, segmap in recs.items():
    content = rec_key[0]
    segs = sorted(segmap)
    i = 0
    while i + args.n_seg <= len(segs):
      run = segs[i:i + args.n_seg]
      if run[-1] - run[0] == args.n_seg - 1:           # truly consecutive
        cands.setdefault(content, []).append((rec_key, run))
        i += args.n_seg                                # non-overlapping
      else:
        i += 1
  # canonical order BEFORE the seeded shuffle -> fully reproducible selection
  for c in cands:
    cands[c].sort(key=lambda rc: (rc[0], rc[1][0]))
    random.shuffle(cands[c])
  contents = sorted(cands)
  random.shuffle(contents)
  total_cands = sum(len(v) for v in cands.values())

  out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
  existing = {p.stem for p in out.glob('*.wav')}
  print('{} segments | {} recordings | {} unique contents | {} candidate triples | target {} | present {}'.format(
      len(files), len(recs), len(contents), total_cands, args.n, len(existing)))

  written, errors = 0, 0
  pbar = tqdm(total=min(args.n - len(existing), total_cands), unit='clip')
  rnd = 0
  while len(existing) + written < args.n:
    progress = False
    for c in contents:                                 # round-robin: one per content per round
      if len(existing) + written >= args.n:
        break
      if rnd >= len(cands[c]):
        continue
      rec_key, run = cands[c][rnd]
      name = '-'.join(rec_key[0]) + '-' + rec_key[1] + '-s{}'.format(run[0])
      if name in existing:
        continue
      try:
        ys, native = [], None
        for seg in run:
          y, sr = sf.read(str(recs[rec_key][seg]), dtype='float32')
          native = sr
          ys.append(y.mean(1) if y.ndim > 1 else y)
        y = np.concatenate(ys)
        if native != args.sr:
          import librosa
          y = librosa.resample(y, orig_sr=native, target_sr=args.sr)
        w = args.n_seg * args.sr
        y = y[:w] if len(y) >= w else np.pad(y, (0, w - len(y)))
        sf.write(out / (name + '.wav'), y.astype('float32'), args.sr, subtype='PCM_16')
        written += 1; progress = True; pbar.update(1)
      except Exception:
        errors += 1
    if not progress:
      break
    rnd += 1
  pbar.close()
  print('wrote {} clips -> {} | total {} ({} errors)'.format(written, out, len(existing) + written, errors))


if __name__ == '__main__':
  main()
