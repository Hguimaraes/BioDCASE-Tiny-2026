# --
# generate a distillation alpha/temperature sweep on the winning no-aug base
#
# the ablation showed pure Perch distillation (no augmentation) is best; this
# sweeps the KD weight (alpha) and temperature (T) around that point to see if
# leaning harder on the teacher helps. writes to experiments/configs/distill_sweep/.
#
#   python experiments/make_distill_sweep.py
#   # then on the cluster:
#   bash cluster/run_ablations.sh 'experiments/configs/distill_sweep/*.yaml'

import copy
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASE = ROOT / 'experiments' / 'configs' / 'abl_distill_no_aug.yaml'   # no-aug distillation base
OUT = ROOT / 'experiments' / 'configs' / 'distill_sweep'

# teacher is strong -> probe higher KD weight; T around the known-good 3.0
ALPHAS = [0.7, 0.9, 1.0]
TEMPERATURES = [2.0, 4.0]


def main():
  base = yaml.safe_load(open(BASE))
  base['skip_deployment_flag'] = True   # experiments never flash the board
  OUT.mkdir(parents=True, exist_ok=True)

  for alpha in ALPHAS:
    for T in TEMPERATURES:
      cfg = copy.deepcopy(base)
      d = cfg['pytorch_framework']['training_recipe']['distillation']
      d['alpha'], d['temperature'] = alpha, T
      tag = 'distill_a{:02d}_T{:g}'.format(int(alpha * 10), T)
      cfg['pytorch_framework']['experiment'].update(
        name=tag.replace('_', '-'), track='C1',
        notes='Distillation sweep (no aug): alpha={}, T={}'.format(alpha, T))
      path = OUT / '{}.yaml'.format(tag)
      yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False, default_flow_style=False)
      print('wrote {} | alpha={} T={}'.format(path.name, alpha, T))


if __name__ == '__main__':
  main()
