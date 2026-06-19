# --
# PCEN-featurize the kept extra clips into a cache_pcen-format cache, using the
# EXACT same front-end as the student's training cache (feature_handler with the
# pcen_distill.yaml params) so extra features are interchangeable with cache_pcen.
#
# Reads kept_clips.txt manifest(s) of 3 s / 24 kHz wav paths, writes per-clip
# <out-dir>/<class>/<stem>.npz with key 'x' = (1,40,133) float32 in [0,1].
#
#   .venv/bin/python experiments/data/featurize_extra.py \
#       --manifest /home/hguimaraes/datasets/extra/xc/kept_clips.txt \
#                  /home/hguimaraes/datasets/extra/background/kept_clips.txt \
#       --out-dir output/02_features/cache_pcen_extra

import sys
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

try:
  from tqdm import tqdm
except ImportError:
  def tqdm(it, **k): return it

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from feature_handler import FeatureHandler

# the cache_pcen front-end, verbatim from experiments/configs/pcen_distill.yaml
PCEN_CFG = dict(
  feature_type='pcen_mel', target_sample_rate=24000, window_len=4096, window_stride=512,
  mel_n_channels=40, mel_low_hz=50, mel_high_hz=7500, transpose_features_extracted=True,
  normalize_features=True, to_float=True, add_channel_dimension=True,
  add_batch_dimension=False, channel_dimension_at_end=False,
)


def main():
  ap = argparse.ArgumentParser(description='PCEN-featurize kept extra clips (cache_pcen-compatible).')
  ap.add_argument('--manifest', nargs='+', required=True, help='kept_clips.txt manifest(s) of clip wav paths')
  ap.add_argument('--out-dir', required=True, help='cache dir: <class>/<stem>.npz with x=(1,40,133)')
  args = ap.parse_args()

  fh = FeatureHandler(**PCEN_CFG)
  paths = []
  for m in args.manifest:
    paths += [Path(l.strip()) for l in open(m) if l.strip()]
  out = Path(args.out_dir)
  print('featurizing {} clips -> {}'.format(len(paths), out))

  written, skipped, errors = 0, 0, 0
  for p in tqdm(paths, unit='clip'):
    dst = out / p.parent.name / (p.stem + '.npz')         # class = clip's parent folder
    if dst.exists():
      skipped += 1; continue
    try:
      wav, _ = sf.read(str(p), dtype='float32')
      if wav.ndim > 1:
        wav = wav.mean(1)
      feat = fh.extract(wav).astype(np.float32)           # (1, 40, 133), min-max normalized
    except Exception as e:
      errors += 1
      if errors <= 5: print('  skip {} ({})'.format(p.name, e))
      continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, x=feat)
    written += 1

  print('wrote {} | skipped(existing) {} | errors {} | shape e.g. (1,40,133)'.format(written, skipped, errors))


if __name__ == '__main__':
  main()
