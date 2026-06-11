# --
# generate Track C2 configs: MSAB-FiLM student vs no-FiLM control
#
# C2 tests the BioME idea (inject modulation-spectrum features via FiLM) on the
# tiny student. To isolate the FiLM/MSAB contribution we generate a matched
# pair at the SAME base architecture and the SAME distillation recipe:
#   - c2_film   : SlimCNNFiLM  (MSAB-FiLM conditioning)
#   - c2_nofilm : SlimCNN      (identical base arch, no context) -- the control
# Both distill from Perch with augmentation off (the winning recipe).
#
#   python experiments/make_C2_configs.py
#   bash cluster/run_ablations.sh 'experiments/configs/C2/*.yaml'

import copy
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASE = ROOT / 'experiments' / 'configs' / 'abl_distill_no_aug.yaml'   # distill, no aug
OUT = ROOT / 'experiments' / 'configs' / 'C2'
MSAB_DIR = './experiments/features/msab/mss_nfft256'

# shared base architecture for both arms (= SlimCNN default)
ARCH = {'stem_ch': 24, 'block_widths': [48, 64, 96, 128], 'block_strides': [2, 2, 2, 1],
        'head_dim': 64, 'dropout': 0.1}


def main():
  base = yaml.safe_load(open(BASE))
  base['skip_deployment_flag'] = True
  OUT.mkdir(parents=True, exist_ok=True)

  # arm 1: MSAB-FiLM student
  film = copy.deepcopy(base)
  pf = film['pytorch_framework']
  pf['model']['attr'] = 'SlimCNNFiLM'
  pf['model']['kwargs'] = {**pf['model']['kwargs'], **ARCH, 'ctx_dim': 258, 'ctx_proj_dim': 32}
  pf['training_recipe']['context'] = {'msab_dir': MSAB_DIR, 'train_split': 'Train', 'eval_split': 'Validation'}
  pf['experiment'].update(name='c2-film', track='C2', notes='C2: SlimCNNFiLM (MSAB-FiLM) + Perch distillation, no aug')

  # arm 2: no-FiLM control (same base arch + same distillation)
  ctrl = copy.deepcopy(base)
  pf = ctrl['pytorch_framework']
  pf['model']['attr'] = 'SlimCNN'
  pf['model']['kwargs'] = {**pf['model']['kwargs'], **ARCH}
  pf['experiment'].update(name='c2-nofilm', track='C2', notes='C2 control: SlimCNN (no FiLM) + Perch distillation, no aug')

  for tag, cfg in [('c2_film', film), ('c2_nofilm', ctrl)]:
    path = OUT / '{}.yaml'.format(tag)
    yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False, default_flow_style=False)
    print('wrote {} | attr={}'.format(path.name, cfg['pytorch_framework']['model']['attr']))


if __name__ == '__main__':
  main()
