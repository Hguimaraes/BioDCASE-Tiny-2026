"""Run mel plus MSS side-channel experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault(
  "MPLCONFIGDIR",
  str(Path(__file__).resolve().parents[1] / "output/matplotlib"),
)

import numpy as np
import soundfile
import torch
import yaml

from experiments import runtime
from experiments.features.modulation_spectrum import ModulationSpectrum2DFeatureHandler
from experiments.mel_mss_datamodule import DatamoduleMelMSS
from experiments.run_baseline import resolve_dataset_root
from experiments.run_mss2d import tee_to_file
from feature_handler import FeatureHandler
from pipeline_pytorch.model_training import pytorch_model_taining


REQUIRED_MODULES = ["numpy", "soundfile", "sklearn", "torch", "torchsummary", "yaml"]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default="experiments/configs/mel_mss_sidechannel.yaml",
    help="Mel+MSS experiment YAML config.",
  )
  parser.add_argument(
    "--mode",
    choices=["preflight", "feature-smoke", "model-smoke", "train"],
    default="preflight",
    help="Which experiment stage to run.",
  )
  parser.add_argument(
    "--dataset-root",
    default=None,
    help="Override dataset root. Parent dirs with one nested Train/Validation child are auto-resolved.",
  )
  parser.add_argument(
    "--num-epochs",
    type=int,
    default=None,
    help="Override pytorch_framework.model_training.num_epochs.",
  )
  return parser.parse_args()


def load_config(path: str | Path, dataset_root: str | None, num_epochs: int | None):
  cfg = runtime.expand_config(runtime.load_yaml(runtime.resolve_project_path(path)))
  base_path = runtime.resolve_project_path(cfg.get("base_config", "./config.yaml"))
  training_cfg = runtime.expand_config(runtime.load_yaml(base_path))
  training_cfg = runtime.deep_update(training_cfg, cfg.get("training_config_overrides", {}))

  root = dataset_root or cfg.get("dataset", {}).get("root_path")
  resolved_root = resolve_dataset_root(root)
  if resolved_root is not None:
    training_cfg["datamodule"]["dataset"]["root_path"] = str(resolved_root)
    cfg.setdefault("dataset", {})["root_path"] = str(resolved_root)

  if num_epochs is not None:
    training_cfg["pytorch_framework"]["model_training"]["num_epochs"] = num_epochs

  cfg["training_config"] = training_cfg
  return cfg


def module_status() -> dict[str, str]:
  status = {}
  for module in REQUIRED_MODULES:
    try:
      importlib.import_module(module)
      status[module] = "ok"
    except Exception as exc:
      status[module] = f"missing: {exc.__class__.__name__}: {exc}"
  return status


def ensure_modules() -> None:
  missing = {k: v for k, v in module_status().items() if v != "ok"}
  if missing:
    joined = "\n".join(f"  - {m}: {s}" for m, s in missing.items())
    raise RuntimeError(f"Missing dependencies:\n{joined}")


def first_dataset_wav(dataset_root: str | None) -> Path:
  if dataset_root:
    dataset_path = resolve_dataset_root(dataset_root)
    if dataset_path:
      files = sorted(dataset_path.glob("**/*.wav"))
      if files:
        return files[0]
  return runtime.PROJECT_ROOT / "submission/test_wav_files/Background/BioDCASE26_TEST_0001_Background.wav"


def feature_handlers(config: dict[str, Any]):
  datamodule_cfg = config["training_config"]["datamodule"]
  mel_handler = FeatureHandler(
    **{
      **datamodule_cfg["feature_extraction"],
      **datamodule_cfg["feature_handler_add_kwargs"],
    }
  )
  mss_handler = ModulationSpectrum2DFeatureHandler(
    **{
      **datamodule_cfg["mss_feature_extraction"],
      **datamodule_cfg.get("mss_feature_handler_add_kwargs", {}),
    }
  )
  return mel_handler, mss_handler


def extract_one_feature(config: dict[str, Any]):
  wav_path = first_dataset_wav(config.get("dataset", {}).get("root_path"))
  wav, sample_rate = soundfile.read(wav_path)
  mel_handler, mss_handler = feature_handlers(config)
  mel = mel_handler.extract(wav).astype(np.float32)
  mss = mss_handler.extract(wav).astype(np.float32)
  features = np.concatenate([mel.flatten(), mss.flatten()]).astype(np.float32)
  return features, mel, mss, wav_path, sample_rate


def preflight(config: dict[str, Any]) -> dict[str, Any]:
  dataset_root = config.get("dataset", {}).get("root_path")
  dataset_path = resolve_dataset_root(dataset_root)
  training_cfg = config["training_config"]
  return {
    "module_status": module_status(),
    "dataset_root": str(dataset_path) if dataset_path else None,
    "dataset_exists": dataset_path.is_dir() if dataset_path else False,
    "cache_id": training_cfg["datamodule"]["caching"]["cache_id"],
    "model": training_cfg["pytorch_framework"]["model"],
    "mel_feature_extraction": training_cfg["datamodule"]["feature_extraction"],
    "mss_feature_extraction": training_cfg["datamodule"]["mss_feature_extraction"],
  }


def feature_smoke(config: dict[str, Any]) -> dict[str, Any]:
  ensure_modules()
  features, mel, mss, wav_path, sample_rate = extract_one_feature(config)
  return {
    "wav_path": str(wav_path),
    "sample_rate": int(sample_rate),
    "feature_shape": list(features.shape),
    "mel_shape": list(mel.shape),
    "mss_shape": list(mss.shape),
    "mel_size": int(mel.size),
    "mss_size": int(mss.size),
    "split_index": int(mel.size),
    "mel_min": float(mel.min()),
    "mel_max": float(mel.max()),
    "mss_mean": float(mss.mean()),
    "mss_std": float(mss.std()),
  }


def model_smoke(config: dict[str, Any]) -> dict[str, Any]:
  ensure_modules()
  features, mel, mss, wav_path, sample_rate = extract_one_feature(config)
  model_cfg = config["training_config"]["pytorch_framework"]["model"]
  model_class = getattr(importlib.import_module(model_cfg["module"]), model_cfg["attr"])
  model = model_class(
    *model_cfg.get("args", []),
    **{
      **model_cfg.get("kwargs", {}),
      "input_shape": tuple(features.shape),
      "num_classes": 11,
      "is_inference_model": True,
      "device": {"use_cpu": True, "device_name": "cpu"},
    },
  )
  x = torch.from_numpy(features).unsqueeze(0).to(dtype=torch.float32)
  with torch.no_grad():
    y = model(x)
  return {
    "wav_path": str(wav_path),
    "sample_rate": int(sample_rate),
    "feature_shape": list(features.shape),
    "mel_shape": list(mel.shape),
    "mss_shape": list(mss.shape),
    "logit_shape": list(y.shape),
    "num_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
  }


def training_log_header(config: dict[str, Any], args: argparse.Namespace, run_dir: Path):
  training_cfg = config["training_config"]
  header = {
    "started_at": dt.datetime.now().isoformat(timespec="seconds"),
    "command": " ".join(sys.argv),
    "mode": args.mode,
    "run_dir": str(run_dir),
    "git": {
      "branch": runtime.git_branch(),
      "commit": runtime.git_commit(),
      "dirty": runtime.git_dirty(),
    },
    "host": {
      "platform": platform.platform(),
      "python": platform.python_version(),
      "torch": str(torch.__version__),
      "cuda_available": torch.cuda.is_available(),
      "cuda_device_count": torch.cuda.device_count(),
      "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
    "dataset": {
      "root_path": training_cfg["datamodule"]["dataset"]["root_path"],
      "cache_id": training_cfg["datamodule"]["caching"]["cache_id"],
      "intermediate_id": training_cfg["datamodule"]["intermediate"]["intermediate_id"],
    },
    "mel_feature_extraction": training_cfg["datamodule"]["feature_extraction"],
    "mss_feature_extraction": training_cfg["datamodule"]["mss_feature_extraction"],
    "model": training_cfg["pytorch_framework"]["model"],
    "training": training_cfg["pytorch_framework"]["model_training"],
    "dataloaders": {
      "train": training_cfg["pytorch_framework"]["dataloader_train_kwargs"],
      "validation_test": training_cfg["pytorch_framework"]["dataloader_validation_and_test_kwargs"],
    },
  }
  print("=== Mel+MSS Training Run ===")
  print(yaml.dump(header, default_flow_style=False, sort_keys=False))
  print("=== Terminal Output ===")


def training_log_footer(results: dict[str, Any], run_dir: Path):
  print("=== Training Summary ===")
  print(yaml.dump(results, default_flow_style=False, sort_keys=False))
  print(f"Run YAML: {run_dir / 'run.yaml'}")
  print(f"Feedback log: {run_dir / 'train.log'}")


def train(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
  ensure_modules()
  training_cfg = config["training_config"]
  training_cfg["pytorch_framework"]["model"]["kwargs"]["save_path"] = str(run_dir / "models")

  datamodule_train = DatamoduleMelMSS(training_cfg["datamodule"], load_set_on_init="train")
  datamodule_validation = DatamoduleMelMSS(training_cfg["datamodule"], load_set_on_init="validation")
  datamodule_test = DatamoduleMelMSS(training_cfg["datamodule"], load_set_on_init="test")
  model = pytorch_model_taining(
    training_cfg["pytorch_framework"],
    datamodule_train,
    datamodule_validation,
    datamodule_test,
  )
  cache_info = datamodule_train.get_cache_info()
  return {
    "input_shape": list(datamodule_train.get_feature_shape_at_load()),
    "mel_shape": cache_info.get("mel_shape"),
    "mss_shape": cache_info.get("mss_shape"),
    "split_index": cache_info.get("split_index"),
    "num_classes": len(datamodule_train.get_label_dict()),
    "dataset_lengths": {
      "train": len(datamodule_train),
      "validation": len(datamodule_validation),
      "test": len(datamodule_test),
    },
    "model_path": str(model.get_model_file_path()),
    "tflite_path": str(model.get_tflite_model_file_path()),
    "num_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    "training_history": getattr(model, "training_history", []),
    "final_epoch": getattr(model, "training_history", [])[-1] if getattr(model, "training_history", []) else None,
    "test_metrics": getattr(model, "test_metrics", {}),
    "best_checkpoints": getattr(model, "best_checkpoints", {}),
  }


def write_run(config: dict[str, Any], mode: str, results: dict[str, Any], run_dir=None) -> Path:
  run_dir = run_dir or runtime.make_run_dir(config, mode)
  record = runtime.run_metadata(config, mode)
  record["results"] = results
  runtime.dump_yaml(record, run_dir / "run.yaml")

  row = {
    "timestamp": record["timestamp"],
    "experiment_id": config["experiment"]["id"],
    "mode": mode,
    "branch": record["git"]["branch"],
    "commit": record["git"]["commit"],
    "run_dir": str(run_dir),
    "mel_mss.feature_shape": results.get("feature_shape") or results.get("input_shape"),
    "mel_mss.mel_shape": results.get("mel_shape"),
    "mel_mss.mss_shape": results.get("mss_shape"),
    "mel_mss.num_params": results.get("num_params"),
    "mel_mss.final_validation_accuracy": (
      results.get("final_epoch") or {}
    ).get("validation_accuracy"),
    "mel_mss.best_validation_accuracy": (
      (results.get("best_checkpoints") or {}).get("best_accuracy") or {}
    ).get("validation_accuracy"),
    "mel_mss.best_accuracy_epoch": (
      (results.get("best_checkpoints") or {}).get("best_accuracy") or {}
    ).get("epoch"),
    "mel_mss.test_accuracy": (
      results.get("test_metrics") or {}
    ).get("test_accuracy"),
    "mel_mss.early_stop_epoch": (
      (results.get("best_checkpoints") or {}).get("early_stopping") or {}
    ).get("stop_epoch"),
    "mel_mss.evaluated_checkpoint": (
      (results.get("test_metrics") or {}).get("evaluated_checkpoint") or {}
    ).get("name"),
    "mel_mss.best_checkpoint_path": (
      (results.get("best_checkpoints") or {}).get("best_accuracy") or {}
    ).get("path"),
    "mel_mss.model_path": results.get("model_path"),
  }
  runtime.append_summary(row, config)
  return run_dir


def main() -> None:
  args = parse_args()
  config = load_config(args.config, args.dataset_root, args.num_epochs)

  if args.mode == "preflight":
    results = preflight(config)
    run_dir = write_run(config, args.mode, results)
  elif args.mode == "feature-smoke":
    results = feature_smoke(config)
    run_dir = write_run(config, args.mode, results)
  elif args.mode == "model-smoke":
    results = model_smoke(config)
    run_dir = write_run(config, args.mode, results)
  else:
    run_dir = runtime.make_run_dir(config, args.mode)
    log_path = run_dir / "train.log"
    with tee_to_file(log_path):
      training_log_header(config, args, run_dir)
      try:
        results = train(config, run_dir)
      except Exception as exc:
        failure = {
          "status": "failed",
          "error": repr(exc),
          "traceback": traceback.format_exc(),
          "log_path": str(log_path),
        }
        write_run(config, args.mode, failure, run_dir=run_dir)
        print("=== Training Failed ===")
        traceback.print_exc()
        print(f"Run YAML: {run_dir / 'run.yaml'}")
        print(f"Feedback log: {log_path}")
        raise
      results["log_path"] = str(log_path)
      write_run(config, args.mode, results, run_dir=run_dir)
      training_log_footer(results, run_dir)

  print(f"Wrote run record: {run_dir / 'run.yaml'}")


if __name__ == "__main__":
  main()
