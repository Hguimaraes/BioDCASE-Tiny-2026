# --
# biodcase 2026 tiny ml - main pipeline pytorch

import os
import sys
import yaml
from pathlib import Path

from datamodule import DatamoduleTinyMl
from pipeline_pytorch.paths import MODELS_DIR
from pipeline_pytorch.model_training import pytorch_model_taining
from model_evaluation import model_evaluation
from experiments.run_logger import RunLogger

# NOTE: int8 quantization (model_quantization -> ai-edge-quantizer) and the
# embedded deployment toolchain (embedded_code_generation / esp_monitor_parser
# -> docker, ESP-IDF) are imported lazily below, so a training-only
# environment (e.g. a SLURM cluster node without those packages) can run the
# full train + float-tflite eval without them installed.

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

  # always evaluate the float tflite model
  print("Float tflite evaluation: ")
  metrics = model_evaluation(cfg, datamodule_test, tflite_path)
  if metrics is not None: run_logger.log_metrics(tflite_float_acc=metrics['acc'], tflite_float_auc=metrics['auc'])
  run_logger.log_artifact_size('tflite_float', tflite_path)

  # int8 quantization (optional): needs ai-edge-quantizer, which a
  # training-only cluster node may not have. Skip via BIODCASE_SKIP_QUANTIZATION
  # or gracefully if the package is missing; quantize such candidates locally.
  quantize = cfg['generate_embedded_code']['quantize'] and not os.environ.get('BIODCASE_SKIP_QUANTIZATION')
  if quantize:
    try:
      from model_quantization import model_quantization
    except ImportError as e:
      print("***ai-edge-quantizer not available ({}); skipping int8 quantization (quantize this candidate locally).".format(e))
      quantize = False

  if quantize:
    # TODO fix overwritten file (add quantization path)
    print("Model quantization (model will be overwritten!) ")
    model_quantization(datamodule_test, tflite_path, tflite_path)

    print("Model evaluation after quantization: ")
    metrics = model_evaluation(cfg, datamodule_test, tflite_path)
    if metrics is not None: run_logger.log_metrics(tflite_int8_acc=metrics['acc'], tflite_int8_auc=metrics['auc'])
    run_logger.log_artifact_size('tflite', tflite_path)

  # finalize run record
  run_logger.finalize(status='completed')

  # skip deployment?
  if cfg['skip_deployment_flag']:
    print("\nSkip deployment! For deployment change 'skip_deployment_flag' to 'False' in 'config.yaml'")
    sys.exit()

  # embedded deployment toolchain (docker / ESP-IDF) - imported lazily so a
  # training-only environment does not require these packages
  from embedded_code_generation import run_compile_embedded_src_code, run_create_target_embedded_src_code, run_deploy_embedded_compiled_code
  from biodcase_tiny.embedded.esp_monitor_parser import finalize_monitor_report

  # run generate embedded src code
  run_create_target_embedded_src_code(cfg, tflite_path)

  # compile
  run_compile_embedded_src_code(cfg)

  # deploy
  run_deploy_embedded_compiled_code(cfg)

  # finalize the monitor report yaml
  finalize_monitor_report("pytorch")
