# --
# Perch v2 embedding export (teacher feature extraction)
#
# Runs the frozen Perch v2 encoder over every clip of a dataset split and
# saves the 1536-d embeddings to disk, keyed by wav file stem so they can
# be aligned to the student's training samples across environments.
#
# Runs in the separate .venv-perch (JAX/Keras/TF):
#   .venv-perch/bin/python experiments/perch/export_embeddings.py \
#       --data-root <dataset> --split Train Validation \
#       --out experiments/perch/embeddings --preset perch_v2_cpu
#
# Output per split: <out>/<preset>/<split>.npz with arrays
#   stems (N,), embeddings (N, 1536), labels (N,), label_names (num_classes,)
# and a shared label_dict.yaml.

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import yaml

from pathlib import Path

# perch wants 32 kHz, 5 s windows
PERCH_WINDOW_S = 5.0


def parse_args():
  p = argparse.ArgumentParser(description='Export Perch v2 embeddings for a dataset.')
  p.add_argument('--data-root', required=True, help='dataset root containing split folders')
  p.add_argument('--split', nargs='+', default=['Train', 'Validation'], help='split folder names')
  p.add_argument('--out', default='experiments/perch/embeddings', help='output base dir')
  p.add_argument('--preset', default='perch_v2_cpu', help='perch_hoplite preset name')
  p.add_argument('--file-ext', default='.wav')
  p.add_argument('--limit', type=int, default=0, help='limit clips per split (0 = all, for smoke tests)')
  return p.parse_args()


def resample_to(x, fs_in, fs_out):
  """
  linear resample (dependency-light, adequate for a frozen feature extractor)
  """

  if fs_in == fs_out: return x
  n_out = int(round(len(x) * fs_out / fs_in))
  return np.interp(np.linspace(0, len(x), n_out, endpoint=False), np.arange(len(x)), x).astype(np.float32)


def load_clip(path, target_sr):
  """
  load mono wav, resample to target_sr, pad/trim to one Perch window
  """

  x, fs = sf.read(path, dtype='float32', always_2d=False)
  if x.ndim > 1: x = x.mean(axis=1)
  x = resample_to(x, fs, target_sr)

  # pad / trim to exactly one window so we get a single embedding frame
  win = int(PERCH_WINDOW_S * target_sr)
  if len(x) < win: x = np.pad(x, (0, win - len(x)))
  elif len(x) > win: x = x[:win]

  return x


def discover_clips(split_dir, file_ext):
  """
  yield (label_name, wav_path) for class-subfolder layout
  """

  clips = []
  for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
    for wav in sorted(class_dir.glob('*{}'.format(file_ext))):
      clips.append((class_dir.name, wav))
  return clips


def main():
  args = parse_args()

  # load model
  from perch_hoplite.zoo import model_configs
  print('Loading Perch preset: {} ...'.format(args.preset))
  model = model_configs.load_model_by_name(args.preset)
  target_sr = model.sample_rate
  print('Model loaded | sample_rate: {}'.format(target_sr))

  data_root = Path(args.data_root)
  out_dir = Path(args.out) / args.preset
  out_dir.mkdir(parents=True, exist_ok=True)

  # build a consistent label dict across all requested splits
  all_labels = set()
  split_clips = {}
  for split in args.split:
    clips = discover_clips(data_root / split, args.file_ext)
    if args.limit: clips = clips[:args.limit]
    split_clips[split] = clips
    all_labels.update(name for name, _ in clips)
  label_dict = {name: i for i, name in enumerate(sorted(all_labels))}
  label_names = [name for name, _ in sorted(label_dict.items(), key=lambda kv: kv[1])]
  yaml.safe_dump({'label_dict': label_dict}, open(out_dir / 'label_dict.yaml', 'w'), sort_keys=False)
  print('Label dict ({} classes): {}'.format(len(label_dict), label_dict))

  # export each split
  for split, clips in split_clips.items():
    print('\nExporting split [{}] - {} clips'.format(split, len(clips)))
    stems, embeddings, labels = [], [], []

    for i, (label_name, wav_path) in enumerate(clips):
      x = load_clip(wav_path, target_sr)
      out = model.embed(x)
      emb = np.asarray(out.embeddings)            # (frames, channels, features)
      emb = emb.reshape(-1, emb.shape[-1]).mean(axis=0)  # mean over frames/channels -> (features,)

      stems.append(wav_path.stem)
      embeddings.append(emb.astype(np.float32))
      labels.append(label_dict[label_name])

      if (i + 1) % 100 == 0 or (i + 1) == len(clips):
        print('  {}/{}'.format(i + 1, len(clips)))

    out_path = out_dir / '{}.npz'.format(split)
    np.savez_compressed(
      out_path,
      stems=np.array(stems),
      embeddings=np.stack(embeddings).astype(np.float32),
      labels=np.array(labels, dtype=np.int64),
      label_names=np.array(label_names),
    )
    print('  saved -> {} | embeddings {}'.format(out_path, np.stack(embeddings).shape))

  print('\nDone. Embeddings written under: {}'.format(out_dir))


if __name__ == '__main__':
  main()
