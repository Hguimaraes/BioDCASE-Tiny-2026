# --
# generate ablation configs for the recipe x distillation 2x2
#
# derives self-contained config files from config.yaml with specific knobs
# flipped, so the recipe (Track A) and distillation (Track C1) effects can be
# attributed independently. writes to experiments/configs/.
#
#   python experiments/make_ablation_configs.py
#
# run each on the cluster (optionally sweeping seeds), e.g.:
#   BIODCASE_CONFIG=experiments/configs/abl_recipe_only.yaml \
#   BIODCASE_SEED=1 sbatch cluster/train.sbatch

import copy
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'experiments' / 'configs'


def _no_augmentation(recipe):
  """disable all Track A regularization (-> baseline-style training signal)"""
  recipe['augmentation'] = {**recipe.get('augmentation', {}), 'enabled': False}
  recipe['mixup'] = {'alpha': 0.0, 'p': 0.0}


def main():
  base = yaml.safe_load(open(ROOT / 'config.yaml'))
  base['skip_deployment_flag'] = True   # experiments never flash the board (deploy is a separate manual step)
  OUT.mkdir(parents=True, exist_ok=True)

  variants = {}

  # --- recipe ON, distill OFF : Track A alone ---
  v = copy.deepcopy(base)
  pf = v['pytorch_framework']
  pf['training_recipe']['distillation']['enabled'] = False
  pf['experiment'].update(name='abl-recipe-only', track='A',
                          notes='Ablation: Track A recipe (aug+mixup+smoothing+cosine), no distillation')
  variants['abl_recipe_only'] = v

  # --- recipe OFF, distill ON : pure distillation ---
  v = copy.deepcopy(base)
  pf = v['pytorch_framework']
  _no_augmentation(pf['training_recipe'])
  pf['training_recipe']['distillation']['enabled'] = True
  pf['model']['kwargs']['criterion']['kwargs']['label_smoothing'] = 0.0
  pf['experiment'].update(name='abl-distill-no-aug', track='C1',
                          notes='Ablation: Perch distillation only, no augmentation/mixup/label-smoothing')
  variants['abl_distill_no_aug'] = v

  # --- recipe ON, distill ON : full C1 (kept for seed sweeps / reproducibility) ---
  v = copy.deepcopy(base)
  v['pytorch_framework']['experiment'].update(name='abl-c1-full', track='C1',
                          notes='Full C1 (recipe + distillation) - reproducibility / seed sweep')
  variants['abl_c1_full'] = v

  for name, cfg in variants.items():
    path = OUT / '{}.yaml'.format(name)
    yaml.safe_dump(cfg, open(path, 'w'), sort_keys=False, default_flow_style=False)
    pf = cfg['pytorch_framework']['training_recipe']
    print('wrote {} | aug={} mixup_p={} distill={} smoothing={}'.format(
      path.name, pf['augmentation']['enabled'], pf['mixup']['p'],
      pf['distillation']['enabled'],
      cfg['pytorch_framework']['model']['kwargs']['criterion']['kwargs'].get('label_smoothing')))


if __name__ == '__main__':
  main()
