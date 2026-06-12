# --
# Perch v2 spatial-embedding export (teacher hints for layer-to-layer distillation)
#
# perch_hoplite's embed() discards the SavedModel's `spatial_embedding` output
# (taxonomy_model_tf.py pops it). We call the serving signature directly and
# keep it: the pre-pool B3 feature map, the deepest-stage hint that our slim-B3
# student's /32 map is shaped to match (same spatial size by construction,
# channels differ -> projected at train time).
#
# Runs in .venv-perch (TF):
#   .venv-perch/bin/python experiments/perch/export_spatial.py \
#       --data-root <dataset> --split Train Validation --preset perch_v2_cpu
#
# Output per split: <out>/<preset>/spatial/<split>.npz with arrays
#   stems (N,), spatial (N, C, H, W), labels (N,)

import argparse
import numpy as np
import yaml
from pathlib import Path

# reuse the audio loader / clip discovery from the embedding exporter
from export_embeddings import load_clip, discover_clips


def parse_args():
  p = argparse.ArgumentParser(description='Export Perch v2 spatial embeddings (teacher hints).')
  p.add_argument('--data-root', required=True)
  p.add_argument('--split', nargs='+', default=['Train', 'Validation'])
  p.add_argument('--out', default='experiments/perch/embeddings')
  p.add_argument('--preset', default='perch_v2_cpu')
  p.add_argument('--file-ext', default='.wav')
  p.add_argument('--limit', type=int, default=0)
  return p.parse_args()


def main():
  args = parse_args()
  from perch_hoplite.zoo import model_configs
  print('Loading Perch preset: {} ...'.format(args.preset))
  model = model_configs.load_model_by_name(args.preset)
  target_sr = model.sample_rate
  tfm = getattr(model, 'model', model)
  sig = tfm.signatures['serving_default']
  print('Model loaded | sample_rate {} | signature outputs {}'.format(
      target_sr, list(sig.structured_outputs.keys())))

  data_root = Path(args.data_root)
  out_dir = Path(args.out) / args.preset / 'spatial'
  out_dir.mkdir(parents=True, exist_ok=True)

  all_labels, split_clips = set(), {}
  for split in args.split:
    clips = discover_clips(data_root / split, args.file_ext)
    if args.limit: clips = clips[:args.limit]
    split_clips[split] = clips
    all_labels.update(name for name, _ in clips)
  label_dict = {name: i for i, name in enumerate(sorted(all_labels))}

  import tensorflow as tf
  for split, clips in split_clips.items():
    print('\nExporting spatial [{}] - {} clips'.format(split, len(clips)))
    stems, spatials, labels = [], [], []
    for i, (label_name, wav_path) in enumerate(clips):
      x = load_clip(wav_path, target_sr).astype(np.float32)
      out = sig(inputs=tf.constant(x[np.newaxis, :]))     # (1, samples)
      sp = np.asarray(out['spatial_embedding'])           # (1, ...) teacher B3 map
      sp = sp[0]                                          # drop batch
      stems.append(wav_path.stem)
      spatials.append(sp.astype(np.float32))
      labels.append(label_dict[label_name])
      if i == 0:
        print('  spatial_embedding shape (per clip): {}'.format(sp.shape))
      if (i + 1) % 100 == 0 or (i + 1) == len(clips):
        print('  {}/{}'.format(i + 1, len(clips)))

    out_path = out_dir / '{}.npz'.format(split)
    np.savez_compressed(out_path, stems=np.array(stems),
                        spatial=np.stack(spatials).astype(np.float32),
                        labels=np.array(labels, dtype=np.int64))
    print('  saved -> {} | spatial {}'.format(out_path, np.stack(spatials).shape))

  print('\nDone. Spatial hints under: {}'.format(out_dir))


if __name__ == '__main__':
  main()
