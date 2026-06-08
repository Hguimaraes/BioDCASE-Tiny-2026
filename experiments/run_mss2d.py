"""Run MSS-only TinyCNN experiments."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
  "MPLCONFIGDIR",
  str(Path(__file__).resolve().parents[1] / "output/matplotlib"),
)

import soundfile
import torch

from experiments import runtime
from experiments.features.modulation_spectrum import ModulationSpectrum2DFeatureHandler
from experiments.mss_datamodule import DatamoduleMSS2D
from experiments.run_baseline import resolve_dataset_root
from pipeline_pytorch.model_training import pytorch_model_taining


REQUIRED_MODULES = ["numpy", "soundfile", "sklearn", "torch", "torchsummary", "yaml"]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default="experiments/configs/mss2d_input.yaml",
    help="MSS experiment YAML config.",
  )
  parser.add_argument(
    "--mode",
    choices=["preflight", "feature-smoke", "model-smoke", "train"],
    default="preflight",
    help="Which MSS experiment stage to run.",
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
    "feature_extraction": training_cfg["datamodule"]["feature_extraction"],
  }


def feature_smoke(config: dict[str, Any]) -> dict[str, Any]:
  ensure_modules()
  features, wav_path, sample_rate = extract_one_feature(config)
  return {
    "wav_path": str(wav_path),
    "sample_rate": int(sample_rate),
    "feature_shape": list(features.shape),
    "feature_min": float(features.min()),
    "feature_max": float(features.max()),
    "feature_mean": float(features.mean()),
    "feature_std": float(features.std()),
  }


def extract_one_feature(config: dict[str, Any]):
  wav_path = first_dataset_wav(config.get("dataset", {}).get("root_path"))
  wav, sample_rate = soundfile.read(wav_path)
  feature_cfg = {
    **config["training_config"]["datamodule"]["feature_extraction"],
    **config["training_config"]["datamodule"]["feature_handler_add_kwargs"],
  }
  extractor = ModulationSpectrum2DFeatureHandler(**feature_cfg)
  features = extractor.extract(wav)
  return features, wav_path, sample_rate


def model_smoke(config: dict[str, Any]) -> dict[str, Any]:
  ensure_modules()
  features, wav_path, sample_rate = extract_one_feature(config)
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
    "logit_shape": list(y.shape),
    "num_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
  }


def train(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
  ensure_modules()
  training_cfg = config["training_config"]
  training_cfg["pytorch_framework"]["model"]["kwargs"]["save_path"] = str(run_dir / "models")

  datamodule_train = DatamoduleMSS2D(training_cfg["datamodule"], load_set_on_init="train")
  datamodule_validation = DatamoduleMSS2D(
    training_cfg["datamodule"],
    load_set_on_init="validation",
  )
  datamodule_test = DatamoduleMSS2D(training_cfg["datamodule"], load_set_on_init="test")
  model = pytorch_model_taining(
    training_cfg["pytorch_framework"],
    datamodule_train,
    datamodule_validation,
    datamodule_test,
  )
  return {
    "input_shape": list(datamodule_train.get_feature_shape_at_load()),
    "num_classes": len(datamodule_train.get_label_dict()),
    "dataset_lengths": {
      "train": len(datamodule_train),
      "validation": len(datamodule_validation),
      "test": len(datamodule_test),
    },
    "model_path": str(model.get_model_file_path()),
    "tflite_path": str(model.get_tflite_model_file_path()),
    "num_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
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
    "mss2d.feature_shape": results.get("feature_shape") or results.get("input_shape"),
    "mss2d.num_params": results.get("num_params"),
    "mss2d.model_path": results.get("model_path"),
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
    results = train(config, run_dir)
    write_run(config, args.mode, results, run_dir=run_dir)

  print(f"Wrote run record: {run_dir / 'run.yaml'}")


if __name__ == "__main__":
  main()
