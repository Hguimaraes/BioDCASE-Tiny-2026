# --
# generate Track B configs: SlimCNN student + Perch distillation (no aug)
#
# builds on the winning recipe (pure distillation, augmentation off) and swaps
# the dense-conv baseline for the depthwise-separable SlimCNN at a few sizes.
# writes to experiments/configs/trackB/.
#
#   python experiments/make_trackB_configs.py
#   bash cluster/run_ablations.sh 'experiments/configs/trackB/*.yaml'

import copy
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASE = ROOT / 'experiments' / 'configs' / 'abl_distill_no_aug.yaml'   # distillation, no aug
OUT = ROOT / 'experiments' / 'configs' / 'trackB'

# (name, arch kwargs) - sizes bracket the baseline budget from below
VARIANTS = {
  'slim_default': {'stem_ch': 24, 'block_widths': [48, 64, 96, 128], 'block_strides': [2, 2, 2, 1], 'head_dim': 64, 'dropout': 0.1},
  'slim_wide':    {'stem_ch': 32, 'block_widths': [64, 96, 128, 160], 'block_strides': [2, 2, 2, 1], 'head_dim': 96, 'dropout': 0.1},
  'slim_deep':    {'stem_ch': 24, 'block_widths': [48, 64, 96, 128, 160], 'block_strides': [2, 2, 2, 2, 1], 'head_dim': 64, 'dropout': 0.1},
}


def main():
  base = yaml.safe_load(open(BASE))
  base['skip_deployment_flag'] = True   # experiments never flash the board
  OUT.mkdir(parents=True, exist_ok=True)

  for name, arch in VARIANTS.items():
    cfg = copy.deepcopy(base)
    model = cfg['pytorch_framework']['model']
    model['attr'] = 'SlimCNN'
    # keep device/criterion/optimizer/verbose, add arch kwargs
    model['kwargs'] = {**model['kwargs'], **arch}
    cfg['pytorch_framework']['experiment'].update(
      name='trackB-{}'.format(name.replace('_', '-')), track='B',
      notes='Track B: SlimCNN ({}) + Perch distillation, no aug'.format(name))
    path = OUT / '{}.yaml'.format(name)
    yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False, default_flow_style=False)
    print('wrote {} | attr=SlimCNN {}'.format(path.name, arch))


if __name__ == '__main__':
  main()
