"""Run baseline pretrained-model checks and log results."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from experiments import runtime


REQUIRED_MODULES = {
  "submission-smoke": [
    "numpy",
    "soundfile",
    "scipy",
    "sklearn",
    "torch",
    "torchinfo",
    "ai_edge_litert",
  ],
  "dataset-eval": [
    "numpy",
    "soundfile",
    "scipy",
    "sklearn",
    "torch",
    "ai_edge_litert",
  ],
}


@contextlib.contextmanager
def pushd(path: Path):
  previous = Path.cwd()
  os.chdir(path)
  try:
    yield
  finally:
    os.chdir(previous)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default="experiments/configs/baseline_pretrained.yaml",
    help="Experiment YAML config.",
  )
  parser.add_argument(
    "--mode",
    choices=["preflight", "submission-smoke", "dataset-eval", "all"],
    default="preflight",
    help="Which baseline check to run.",
  )
  parser.add_argument(
    "--dataset-root",
    default=None,
    help="Override datamodule.dataset.root_path for dataset evaluation.",
  )
  return parser.parse_args()


def load_experiment_config(path: str | Path, dataset_root: str | None) -> dict[str, Any]:
  cfg = runtime.expand_config(runtime.load_yaml(runtime.resolve_project_path(path)))
  if dataset_root:
    cfg.setdefault("dataset", {})["root_path"] = dataset_root
  return cfg


def module_status(modules: list[str]) -> dict[str, str]:
  status = {}
  for module in modules:
    try:
      importlib.import_module(module)
      status[module] = "ok"
    except Exception as exc:
      status[module] = f"missing: {exc.__class__.__name__}: {exc}"
  return status


def preflight(config: dict[str, Any]) -> dict[str, Any]:
  modules = sorted({m for values in REQUIRED_MODULES.values() for m in values})
  status = module_status(modules)
  files = {}
  for model in config.get("models", []):
    path = runtime.resolve_project_path(model["path"])
    files[model["name"]] = {
      "path": str(path),
      "exists": path.is_file(),
      "size_bytes": path.stat().st_size if path.is_file() else None,
    }
  dataset_root = config.get("dataset", {}).get("root_path")
  dataset_path = resolve_dataset_root(dataset_root)
  return {
    "module_status": status,
    "models": files,
    "dataset_root": str(dataset_path) if dataset_path else None,
    "dataset_exists": dataset_path.is_dir() if dataset_path else False,
  }


def ensure_modules(mode: str) -> None:
  missing = {
    module: status
    for module, status in module_status(REQUIRED_MODULES[mode]).items()
    if status != "ok"
  }
  if missing:
    joined = "\n".join(f"  - {m}: {s}" for m, s in missing.items())
    raise RuntimeError(f"Missing dependencies for {mode}:\n{joined}")


def configured_path(value: str | None) -> Path | None:
  if not value or "$" in value:
    return None
  return runtime.resolve_project_path(value)


def resolve_dataset_root(value: str | None) -> Path | None:
  dataset_path = configured_path(value)
  if dataset_path is None:
    return None
  if (dataset_path / "Train").is_dir() and (dataset_path / "Validation").is_dir():
    return dataset_path
  if dataset_path.is_dir():
    candidates = [
      child
      for child in dataset_path.iterdir()
      if child.is_dir() and (child / "Train").is_dir() and (child / "Validation").is_dir()
    ]
    if len(candidates) == 1:
      return candidates[0]
  return dataset_path


def run_submission_smoke(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
  ensure_modules("submission-smoke")
  run_dir = runtime.make_run_dir(config, "submission-smoke")
  submission_config_path = runtime.resolve_project_path(config["submission"]["config_path"])
  submission_dir = submission_config_path.parent
  with pushd(submission_dir):
    if str(submission_dir) not in sys.path:
      sys.path.append(str(submission_dir))
    if str(runtime.PROJECT_ROOT) not in sys.path:
      sys.path.append(str(runtime.PROJECT_ROOT))
    from submission_test import run_inference, run_write_final_results

    cfg = yaml.safe_load(submission_config_path.read_text())
    report_dir = run_dir / "submission_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    inference_scores_file = report_dir / "inference_scores.yaml"
    monitor_report_file = report_dir / "monitor_report.yaml"
    submission_results_file = report_dir / "submission_results.yaml"
    cm_plot_file = report_dir / "cm_inference.png"

    tflite_path = run_inference(cfg, inference_scores_file, cm_plot_file)
    if not config.get("submission", {}).get("skip_embedded", True):
      from submission_test import run_embedded

      run_embedded(cfg, tflite_path, monitor_report_file)
    result = run_write_final_results(
      cfg,
      inference_scores_file,
      monitor_report_file,
      submission_results_file,
    )
  return {
    "report_dir": str(report_dir),
    "submission_results": result,
  }, run_dir


def build_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
  dataset_cfg_path = runtime.resolve_project_path(config["dataset"]["config_path"])
  cfg = runtime.load_yaml(dataset_cfg_path)
  dataset_root = config.get("dataset", {}).get("root_path")
  dataset_path = resolve_dataset_root(dataset_root)
  if dataset_path:
    cfg["datamodule"]["dataset"]["root_path"] = str(dataset_path)
  cfg["skip_deployment_flag"] = True
  return cfg


def run_dataset_eval(config: dict[str, Any]) -> dict[str, Any]:
  ensure_modules("dataset-eval")
  import numpy as np
  from scipy.special import softmax
  from sklearn.metrics import accuracy_score, roc_auc_score

  from datamodule import DatamoduleTinyMl
  from model_evaluation import run_model_main

  cfg = build_dataset_config(config)
  datamodule_test = DatamoduleTinyMl(
    cfg["datamodule"],
    load_set_on_init=config.get("dataset", {}).get("split", "test"),
  )
  y_true = datamodule_test.targets
  results = []
  for model in config.get("models", []):
    model_path = runtime.resolve_project_path(model["path"])
    y_pred = run_model_main(cfg, datamodule_test, model_path)
    y_prob = softmax(y_pred, axis=1)
    y_class = np.argmax(y_pred, axis=1)
    results.append(
      {
        "name": model["name"],
        "path": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "top1_accuracy": float(accuracy_score(y_true, y_class)),
        "roc_auc_macro_ovr": float(
          roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        ),
      }
    )
  return {
    "dataset_length": int(len(datamodule_test)),
    "label_dict": datamodule_test.get_label_dict(),
    "models": results,
  }


def write_run(config: dict[str, Any], mode: str, results: dict[str, Any]) -> Path:
  run_dir = runtime.make_run_dir(config, mode)
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
  }
  if isinstance(results.get("models"), list):
    for model in results["models"]:
      prefix = model["name"]
      row[f"{prefix}.top1_accuracy"] = model.get("top1_accuracy")
      row[f"{prefix}.roc_auc_macro_ovr"] = model.get("roc_auc_macro_ovr")
      row[f"{prefix}.size_bytes"] = model.get("size_bytes")
  runtime.append_summary(row, config)
  return run_dir


def main() -> None:
  mpl_config_dir = runtime.resolve_project_path("./output/matplotlib")
  mpl_config_dir.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

  args = parse_args()
  config = load_experiment_config(args.config, args.dataset_root)
  modes = ["submission-smoke", "dataset-eval"] if args.mode == "all" else [args.mode]
  for mode in modes:
    if mode == "preflight":
      results = preflight(config)
    elif mode == "submission-smoke":
      results, run_dir = run_submission_smoke(config)
      # run_submission_smoke needs the run dir before writing nested reports, so log in place.
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
      }
      for key, value in results.get("submission_results", {}).items():
        row[f"submission.{key}"] = value
      runtime.append_summary(
        row,
        config,
      )
      print(f"Wrote run record: {run_dir / 'run.yaml'}")
      continue
    elif mode == "dataset-eval":
      results = run_dataset_eval(config)
    else:
      raise ValueError(mode)
    run_dir = write_run(config, mode, results)
    print(f"Wrote run record: {run_dir / 'run.yaml'}")


if __name__ == "__main__":
  main()
