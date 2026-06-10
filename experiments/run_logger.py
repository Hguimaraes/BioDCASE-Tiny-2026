# --
# experiment run logger
#
# every training run produces a self-contained record folder under
# experiments/runs/<run_id>/ holding the full resolved config, environment
# info (git sha, host, library versions, device), seed, per-epoch metrics
# and final metrics. these folders are small and tracked in git, so runs
# executed on the cluster can be committed there and pulled back locally.

import os
import csv
import random
import socket
import platform
import subprocess
import yaml
import numpy as np

from datetime import datetime, timezone
from pathlib import Path


def set_global_seed(seed):
  """
  seed all relevant random number generators
  """

  import torch

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _git_info():
  """
  git sha, branch and dirty state of current repo (best effort)
  """

  def _run(args):
    try: return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return None

  return {
    'sha': _run(['git', 'rev-parse', 'HEAD']),
    'branch': _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
    'dirty': bool(_run(['git', 'status', '--porcelain'])),
  }


def _env_info():
  """
  environment info (best effort)
  """

  import torch

  return {
    'host': socket.gethostname(),
    'platform': platform.platform(),
    'python': platform.python_version(),
    'torch': torch.__version__,
    'numpy': np.__version__,
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
  }


class RunLogger:
  """
  experiment run logger - one instance per training run
  """

  def __init__(self, cfg_experiment, full_config=None):

    # config with defaults
    self.cfg = {**{'name': 'unnamed', 'track': '-', 'runs_dir': './experiments/runs', 'seed': 42, 'notes': ''}, **(cfg_experiment or {})}

    # run id and folder
    self.run_id = '{}_{}'.format(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'), self.cfg['name'])
    self.run_dir = Path(self.cfg['runs_dir']) / self.run_id
    self.run_dir.mkdir(parents=True, exist_ok=True)

    # record
    self.record = {
      'run_id': self.run_id,
      'name': self.cfg['name'],
      'track': self.cfg['track'],
      'notes': self.cfg['notes'],
      'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
      'seed': self.cfg['seed'],
      'git': _git_info(),
      'env': _env_info(),
      'status': 'running',
      'data': {},
      'model': {},
      'metrics': {},
      'config': full_config,
    }

    # metrics csv
    self.metrics_csv_path = self.run_dir / 'metrics.csv'
    self.metrics_csv_fields = None

    # seed everything
    set_global_seed(self.cfg['seed'])

    # first write
    self.flush()

    # info
    print('RunLogger - logging run to: {}'.format(self.run_dir))


  def log_data_info(self, **kwargs): self.record['data'].update(kwargs); self.flush()
  def log_model_info(self, **kwargs): self.record['model'].update(kwargs); self.flush()
  def log_metrics(self, **kwargs): self.record['metrics'].update(kwargs); self.flush()


  def log_epoch(self, epoch, **metrics):
    """
    append one row of per-epoch metrics to metrics.csv
    """

    row = {'epoch': epoch, **metrics}

    # write header once, keep field order stable
    if self.metrics_csv_fields is None:
      self.metrics_csv_fields = list(row.keys())
      with open(self.metrics_csv_path, 'w', newline='') as f: csv.DictWriter(f, fieldnames=self.metrics_csv_fields).writeheader()

    with open(self.metrics_csv_path, 'a', newline='') as f: csv.DictWriter(f, fieldnames=self.metrics_csv_fields, extrasaction='ignore').writerow(row)


  def log_artifact_size(self, name, file_path):
    """
    record size in bytes of an artifact file (e.g. .pth / .tflite model)
    """

    file_path = Path(file_path)
    if file_path.is_file(): self.record['model']['{}_bytes'.format(name)] = file_path.stat().st_size
    self.flush()


  def finalize(self, status='completed'):
    """
    mark run as finished
    """

    self.record['status'] = status
    self.record['finished_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    self.flush()
    print('RunLogger - run [{}] finalized with status: {}'.format(self.run_id, status))


  def flush(self):
    """
    write run.yaml
    """

    with open(self.run_dir / 'run.yaml', 'w') as f: yaml.safe_dump(_to_plain(self.record), f, sort_keys=False, default_flow_style=False)


def _to_plain(obj):
  """
  reduce arbitrary objects to yaml-safe builtins (e.g. torch.TorchVersion,
  numpy scalars, pathlib paths)
  """

  if isinstance(obj, dict): return {_to_plain(k): _to_plain(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)): return [_to_plain(v) for v in obj]
  if isinstance(obj, bool) or obj is None: return obj
  if isinstance(obj, (int, float)) and type(obj) in (int, float): return obj
  if isinstance(obj, np.integer): return int(obj)
  if isinstance(obj, np.floating): return float(obj)
  if isinstance(obj, str) and type(obj) is str: return obj
  return str(obj)
