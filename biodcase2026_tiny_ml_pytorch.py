# --
# biodcase 2026 tiny ml - main pipeline pytorch

import os
import sys
import yaml
from pathlib import Path

from datamodule import DatamoduleTinyMl
from pipeline_pytorch.paths import MODELS_DIR
from pipeline_pytorch.model_training import pytorch_model_taining
from embedded_code_generation import run_compile_embedded_src_code, run_create_target_embedded_src_code, run_deploy_embedded_compiled_code
from model_evaluation import model_evaluation
from model_quantization import model_quantization
from biodcase_tiny.embedded.esp_monitor_parser import finalize_monitor_report
from experiments.run_logger import RunLogger

if __name__ == '__main__':
  """
  biodcase tiny ml - main pipeline pytorch
  """

  # yaml config file (path overridable for config sweeps)
  cfg = yaml.safe_load(open(os.environ.get('BIODCASE_CONFIG', './config.yaml')))

  # environment overrides (cluster / local portability, see cluster/train.sbatch)
  if os.environ.get('BIODCASE_DATA_ROOT'): cfg['datamodule']['dataset']['root_path'] = os.environ['BIODCASE_DATA_ROOT']
  if os.environ.get('BIODCASE_SKIP_DEPLOYMENT'): cfg['skip_deployment_flag'] = True

  # info
  print("Hello Tiny ML 2026 - pytorch framework, version: {}".format(cfg['version']))

  # run logger (records config, git sha, seed, env and metrics; also seeds all rngs)
  run_logger = RunLogger(cfg['pytorch_framework'].get('experiment', {}), full_config=cfg)

  # load datamodules
  datamodule_train = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='train')
  datamodule_validation = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='validation')
  datamodule_test = DatamoduleTinyMl(cfg['datamodule'], load_set_on_init='test')
  datamodule_train.info()

  # model training and test
  try:
    model = pytorch_model_taining(cfg['pytorch_framework'], datamodule_train, datamodule_validation, datamodule_test, run_logger=run_logger)
  except Exception:
    run_logger.finalize(status='failed')
    raise

  # tflite model
  tflite_path = model.get_tflite_model_file_path()

  # check existance
  if not tflite_path.is_file():
    print("***Your .tflite model could not be found at: {}\nExit!".format(tflite_path))
    run_logger.finalize(status='no_tflite')
    sys.exit()

  # quantize
  if cfg['generate_embedded_code']['quantize']:
    print("Model evaluation before quantization: ")
    metrics = model_evaluation(cfg, datamodule_test, tflite_path)
    if metrics is not None: run_logger.log_metrics(tflite_float_acc=metrics['acc'], tflite_float_auc=metrics['auc'])

    # TODO fix overwritten file (add quantization path)
    print("Model quantization (model will be overwritten!) ")
    model_quantization(datamodule_test, tflite_path, tflite_path)

    print("Model evaluation after quantization: ")

  # evaluation .tflite model
  metrics = model_evaluation(cfg, datamodule_test, tflite_path)
  if metrics is not None: run_logger.log_metrics(tflite_int8_acc=metrics['acc'], tflite_int8_auc=metrics['auc'])

  # tflite artifact size
  run_logger.log_artifact_size('tflite', tflite_path)

  # finalize run record
  run_logger.finalize(status='completed')

  # skip deployment?
  if cfg['skip_deployment_flag']:
    print("\nSkip deployment! For deployment change 'skip_deployment_flag' to 'False' in 'config.yaml'")
    sys.exit()

  # run generate embedded src code
  run_create_target_embedded_src_code(cfg, tflite_path)

  # compile
  run_compile_embedded_src_code(cfg)

  # deploy
  run_deploy_embedded_compiled_code(cfg)

  # finalize the monitor report yaml
  finalize_monitor_report("pytorch")
