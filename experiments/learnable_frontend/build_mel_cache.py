# --
# Build the raw MAGNITUDE MEL-ENERGY cache (pre-PCEN) that the learnable front-end
# (LearnFrontNet / LearnablePCEN) trains on. Same framing as cache_pcen (24 kHz,
# 40 mel, win 4096 / hop 512 -> 133 frames, f 50..7500, power=1.0) but WITHOUT the
# pcen() step -- PCEN is moved into the model so its params can be learned.
#
# Reads the sliced int16 waveforms from output/01_intermediate/intermediate0/<split>/
# <class>/<stem>.npz (key 'x') and writes output/02_features/cache_mel/<split>/<class>/
# <stem>.npz with key 'x' = (40, 133) float32 raw mel energy. Stems match cache_pcen,
# so the Perch teacher logits/embeddings (keyed by stem) still align.
#
#   .venv/bin/python experiments/learnable_frontend/build_mel_cache.py
#   .venv/bin/python experiments/learnable_frontend/build_mel_cache.py --splits Train Validation

import sys, argparse
from pathlib import Path
import numpy as np
import torch

try:
  from tqdm import tqdm
except ImportError:
  def tqdm(it, **k): return it

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from biodcase_tiny.feature_extraction.pcen_mel import PCENMel

INTER = ROOT / 'output' / '01_intermediate' / 'intermediate0'
OUT = ROOT / 'output' / '02_features' / 'cache_mel'
# framing identical to cache_pcen: FeatureHandler's pcen_mel path builds PCENMel with
# f_min=mel_low_hz(50) and f_max=sample_rate/2 (=12000) -- mel_high_hz is IGNORED there
# (feature_handler.py:77). Match that exactly or the mel energies (hence PCEN) diverge.
MEL_KW = dict(sample_rate=24000, window_len=4096, window_stride=512,
              n_mels=40, f_min=50.0, f_max=12000.0, power=1.0)


def main():
  ap = argparse.ArgumentParser(description='Build pre-PCEN mel-energy cache for the learnable front-end.')
  ap.add_argument('--splits', nargs='+', default=['Train', 'Validation'])
  ap.add_argument('--inter', type=Path, default=INTER)
  ap.add_argument('--out', type=Path, default=OUT)
  args = ap.parse_args()

  pm = PCENMel(**MEL_KW)                                  # we use pm.melspec only (skip the pcen step)
  imax = float(np.iinfo(np.int16).max)
  written = skipped = errors = 0
  for split in args.splits:
    files = sorted((args.inter / split).glob('*/*.npz'))
    print('{}: {} clips'.format(split, len(files)))
    for f in tqdm(files, unit='clip'):
      dst = args.out / split / f.parent.name / (f.stem + '.npz')
      if dst.exists():
        skipped += 1; continue
      try:
        d = np.load(f)
        wav = d['x'].astype(np.float32) / imax           # int16 -> [-1, 1]
        with torch.no_grad():
          mel = pm.melspec(torch.from_numpy(wav))        # (40, 133) magnitude mel energy
        mel = mel.cpu().numpy().astype(np.float32)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, x=mel)
        written += 1
      except Exception as e:
        errors += 1
        print('  ! {}: {}'.format(f.name, e))
  print('done: {} written, {} skipped, {} errors -> {}'.format(written, skipped, errors, args.out))


if __name__ == '__main__':
  main()
