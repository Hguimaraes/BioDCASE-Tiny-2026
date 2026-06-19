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
import time
import argparse
import numpy as np
import soundfile as sf
import yaml

from pathlib import Path

try:
  from tqdm import tqdm
  _HAS_TQDM = True
except ImportError:                                 # fall back to periodic ETA prints
  _HAS_TQDM = False

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
  # resumability for long (overnight) runs: flush shards periodically, skip
  # already-embedded clips on restart, merge shards into the final <split>.npz.
  p.add_argument('--shard-size', type=int, default=2000, help='flush a shard every N clips (crash-safety granularity)')
  p.add_argument('--restart', action='store_true', help='ignore/clear existing shards and start the split fresh')
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


def embed_clip(model, x):
  """Perch embedding for one clip, mean-pooled over frames/channels -> (features,)."""
  out = model.embed(x)
  emb = np.asarray(out.embeddings)                  # (frames, channels, features)
  return emb.reshape(-1, emb.shape[-1]).mean(axis=0).astype(np.float32)


def export_split(model, clips, split, out_dir, label_dict, label_names, target_sr, shard_size, restart):
  """Resumable export of one split: embed clips, flush shards every `shard_size`,
  skip clips already in shards on restart, then merge shards -> <split>.npz.

  Crash-safety: at most `shard_size` clips of work is lost on a kill; rerun the
  same command to resume. A bad clip is skipped (counted), never aborts the run.
  """
  final_path = out_dir / '{}.npz'.format(split)
  parts_dir = out_dir / '{}_parts'.format(split)
  if restart and parts_dir.exists():
    for pf in parts_dir.glob('part_*.npz'): pf.unlink()
  parts_dir.mkdir(parents=True, exist_ok=True)

  # resume: collect stems already embedded in existing shards, skip them
  existing = sorted(parts_dir.glob('part_*.npz'))
  done = set()
  for pf in existing:
    done.update(str(s) for s in np.load(pf, allow_pickle=True)['stems'])
  todo = [(ln, wp) for ln, wp in clips if wp.stem not in done]
  print('\nExporting split [{}] - {} clips ({} already done, {} to do)'.format(
      split, len(clips), len(done), len(todo)))

  part_idx = len(existing)
  buf_s, buf_e, buf_l = [], [], []

  def flush():
    nonlocal part_idx, buf_s, buf_e, buf_l
    if not buf_s: return
    np.savez_compressed(parts_dir / 'part_{:05d}.npz'.format(part_idx),
                        stems=np.array(buf_s), embeddings=np.stack(buf_e).astype(np.float32),
                        labels=np.array(buf_l, dtype=np.int64))
    part_idx += 1; buf_s, buf_e, buf_l = [], [], []

  t0, errors = time.time(), 0
  loop = tqdm(todo, desc='[{}]'.format(split), unit='clip', smoothing=0.05) if _HAS_TQDM else todo
  for i, (label_name, wav_path) in enumerate(loop):
    try:
      buf_e.append(embed_clip(model, load_clip(wav_path, target_sr)))
    except Exception as ex:                              # skip a bad clip, keep going
      errors += 1
      if errors <= 5: print('  skip {} ({})'.format(wav_path.name, ex))
      continue
    buf_s.append(wav_path.stem); buf_l.append(label_dict[label_name])
    if len(buf_s) >= shard_size: flush()
    if _HAS_TQDM:
      if errors: loop.set_postfix(err=errors, refresh=False)
    elif (i + 1) % 200 == 0 or (i + 1) == len(todo):
      rate = (i + 1) / max(1e-9, time.time() - t0)
      print('  {}/{} | {:.1f} clips/s | ETA {:.0f} min | {} errors'.format(
          i + 1, len(todo), rate, (len(todo) - (i + 1)) / max(1e-9, rate) / 60, errors))
  flush()

  # merge all shards -> the final consumer-facing single npz (atomic via .tmp)
  parts = sorted(parts_dir.glob('part_*.npz'))
  if not parts:
    print('  no shards to merge for [{}]'.format(split)); return
  S, E, L = [], [], []
  for pf in parts:
    d = np.load(pf, allow_pickle=True); S.append(d['stems']); E.append(d['embeddings']); L.append(d['labels'])
  stems, embeddings, labels = np.concatenate(S), np.concatenate(E).astype(np.float32), np.concatenate(L)
  tmp = final_path.with_name(final_path.name + '.tmp.npz')
  np.savez_compressed(tmp, stems=stems, embeddings=embeddings, labels=labels, label_names=np.array(label_names))
  tmp.replace(final_path)
  print('  merged {} shards -> {} | embeddings {} | {} errors (delete {}/ to reclaim disk)'.format(
      len(parts), final_path, embeddings.shape, errors, parts_dir.name))


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

  # export each split (resumable: shards + skip-already-done + merge)
  for split, clips in split_clips.items():
    export_split(model, clips, split, out_dir, label_dict, label_names, target_sr,
                 args.shard_size, args.restart)

  print('\nDone. Embeddings written under: {}'.format(out_dir))


if __name__ == '__main__':
  main()
