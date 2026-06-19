# --
# generate embedding-distillation + recipe-bundle configs on the PCEN-Baseline.
#
# Builds on the current best (PCEN-mel + Perch logit distillation, 0.6497/0.9296)
# by adding, on top of the SAME deployable Baseline:
#   #1  embedding (feature) distillation -- regress the teacher's 1536-d Perch
#       embedding from the student's GAP descriptor (train-only projection head).
#   Tier-2 bundle:
#       - weight EMA (eval + checkpoint on the averaged weights)
#       - class-imbalance handling on the CE term (logit-adjust / weighted)
#       - a small alpha/T probe that leans harder on the strong teacher.
#
# All variants keep the exported model byte-for-byte identical to the Baseline;
# only the training signal changes. Needs the teacher embeddings present at
# experiments/perch/embeddings/perch_v2_cpu/Train.npz (gitignored -> copy or
# regenerate via experiments/perch/export_embeddings.py).
#
#   python experiments/make_embed_distill_configs.py
#   # then, e.g. (cluster):
#   bash cluster/run_ablations.sh 'experiments/configs/pcen_embed.yaml' '1 2 3'
#   bash cluster/run_ablations.sh 'experiments/configs/embed_sweep/*.yaml' '42'

import copy
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASE = ROOT / 'experiments' / 'configs' / 'pcen_distill.yaml'        # PCEN + logit distill
EMB_DIR = './experiments/perch/embeddings/perch_v2_cpu'              # holds <split>.npz embeddings

# (file, dir, embed_on, beta, ema, cb_mode, alpha, T, name, note)
#   beta   = weight on the embedding hint loss
#   cb_mode= none | logit_adjust | weighted   (CE-term class imbalance handling)
VARIANTS = [
  # headline candidate: embedding distill + EMA (lives at configs/ root)
  ('pcen_embed.yaml', '.', True, 1.0, True, 'none', 0.5, 3.0,
   'pcen-embed', 'PCEN + logit + embedding distillation + EMA'),

  # --- embed_sweep/: ablations around the headline ---
  ('embed_sweep/embed_b05.yaml', 'embed_sweep', True, 0.5, True, 'none', 0.5, 3.0,
   'embed-b05', 'embedding hint weight 0.5 + EMA'),
  ('embed_sweep/embed_b20.yaml', 'embed_sweep', True, 2.0, True, 'none', 0.5, 3.0,
   'embed-b20', 'embedding hint weight 2.0 + EMA'),
  ('embed_sweep/embed_b10_noema.yaml', 'embed_sweep', True, 1.0, False, 'none', 0.5, 3.0,
   'embed-b10-noema', 'embedding hint weight 1.0, EMA off (isolates EMA)'),
  ('embed_sweep/embed_b10_logitadj.yaml', 'embed_sweep', True, 1.0, True, 'logit_adjust', 0.5, 3.0,
   'embed-b10-logitadj', 'embedding + EMA + logit-adjusted CE'),
  ('embed_sweep/embed_b10_weighted.yaml', 'embed_sweep', True, 1.0, True, 'weighted', 0.5, 3.0,
   'embed-b10-weighted', 'embedding + EMA + class-weighted CE'),
  ('embed_sweep/embed_b10_a07T3.yaml', 'embed_sweep', True, 1.0, True, 'none', 0.7, 3.0,
   'embed-b10-a07T3', 'embedding + EMA + lean harder on teacher (alpha 0.7)'),
  ('embed_sweep/embed_b10_a07T4.yaml', 'embed_sweep', True, 1.0, True, 'none', 0.7, 4.0,
   'embed-b10-a07T4', 'embedding + EMA + alpha 0.7, T 4'),

  # EMA-only control (no embedding distill) to isolate the Tier-2 bundle
  ('embed_sweep/ema_only.yaml', 'embed_sweep', False, 1.0, True, 'none', 0.5, 3.0,
   'ema-only', 'best recipe + EMA only (no embedding distill)'),
]


def main():
  base = yaml.safe_load(open(BASE))
  for fname, subdir, embed_on, beta, ema, cb_mode, alpha, T, name, note in VARIANTS:
    cfg = copy.deepcopy(base)
    cfg['skip_deployment_flag'] = True

    recipe = cfg['pytorch_framework']['training_recipe']
    d = recipe['distillation']
    d['alpha'], d['temperature'] = alpha, T
    d['embed'] = {'enabled': embed_on, 'emb_dir': EMB_DIR, 'weight': beta,
                  'mse_weight': 1.0, 'cos_weight': 1.0, 'standardize': True}
    recipe['ema'] = {'enabled': ema, 'decay': 0.999}
    recipe['class_balance'] = {'mode': cb_mode, 'tau': 1.0}

    cfg['pytorch_framework']['experiment'].update(
      name=name, track='D2',
      notes='Track D2 (embed + recipe bundle): {}'.format(note))

    out = ROOT / 'experiments' / 'configs' / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False, default_flow_style=False)
    print('wrote {:42s} | embed={} beta={} ema={} cb={} alpha={} T={}'.format(
      fname, embed_on, beta, ema, cb_mode, alpha, T))


if __name__ == '__main__':
  main()
