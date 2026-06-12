# --
# generate the Track D config: PCEN-mel front-end vs the int log-mel baseline.
#
# Takes the winning recipe (Baseline + Perch distillation, no aug) and changes
# ONLY the student front-end to a Perch-like PCEN-mel (same framing/mel-count),
# writing features to a separate cache so it coexists with the log-mel cache.
# The log-mel control is just the unchanged abl_distill_no_aug.yaml.
#
#   python experiments/make_pcen_config.py
#   # then, e.g.:
#   bash cluster/run_ablations.sh 'experiments/configs/abl_distill_no_aug.yaml experiments/configs/pcen_distill.yaml'

import copy
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASE = ROOT / 'experiments' / 'configs' / 'abl_distill_no_aug.yaml'   # log-mel + distill


# (file, cache_id, n_mels, run-name) variants. 40 mel = matched-framing A/B vs
# log-mel; 80 mel pushes resolution toward Perch's 128 (input H doubles -> ~2x
# conv MACs, params ~unchanged since the head is fed by channel count not H/W).
VARIANTS = [
  ('pcen_distill.yaml',       'cache_pcen',   40, 'pcen-distill'),
  ('pcen_distill_80mel.yaml', 'cache_pcen80', 80, 'pcen-distill-80mel'),
]


def main():
  for fname, cache_id, n_mels, name in VARIANTS:
    cfg = yaml.safe_load(open(BASE))
    cfg['skip_deployment_flag'] = True

    dm = cfg['datamodule']
    # swap the front-end to PCEN-mel (matched framing; freq range -> Nyquist)
    dm['feature_handler_add_kwargs']['feature_type'] = 'pcen_mel'
    dm['feature_handler_add_kwargs']['mel_low_hz'] = 50      # Perch lower edge
    dm['feature_extraction']['mel_low_hz'] = 50
    dm['feature_extraction']['mel_n_channels'] = n_mels
    # separate cache per resolution so features do not clobber each other
    dm['caching']['cache_id'] = cache_id

    pf = cfg['pytorch_framework']
    pf['experiment'].update(
      name=name, track='D',
      notes='Track D: PCEN-mel ({} mel) + Baseline + Perch distillation, no aug'.format(n_mels))

    out = ROOT / 'experiments' / 'configs' / fname
    yaml.safe_dump(cfg, open(out, 'w'), sort_keys=False, default_flow_style=False)
    print('wrote {} | n_mels={} | cache_id={}'.format(fname, n_mels, cache_id))


if __name__ == '__main__':
  main()
