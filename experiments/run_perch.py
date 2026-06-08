"""Preflight Perch 2.0 / Perch-Hoplite evaluation on BioDCASE labels."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from experiments import runtime
from experiments.run_baseline import resolve_dataset_root


MPL_CONFIG_DIR = runtime.PROJECT_ROOT / "output/matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

REQUIRED_MODULES = [
  "jax",
  "flax",
  "tensorflow",
  "tf_keras",
  "ml_collections",
  "etils",
  "apache_beam",
  "aqt",
  "perch_hoplite",
]

PERCH_MODULES = [
  "chirp",
  "chirp.models.perch_2",
  "chirp.inference.embed_lib",
]

HOPLITE_MODULES = [
  "perch_hoplite.zoo.model_configs",
  "perch_hoplite.zoo.zoo_interface",
]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--config",
    default="experiments/configs/perch2_eval.yaml",
    help="Perch evaluation YAML config.",
  )
  parser.add_argument(
    "--mode",
    choices=["preflight", "label-map", "hoplite-smoke"],
    default="preflight",
    help="Which Perch setup stage to run.",
  )
  parser.add_argument(
    "--dataset-root",
    default=None,
    help="Override dataset root. Parent dirs with one nested Train/Validation child are auto-resolved.",
  )
  parser.add_argument(
    "--perch-repo",
    default=None,
    help="Override local google-research/perch checkout.",
  )
  parser.add_argument(
    "--model-choice",
    default=None,
    help="Override Perch-Hoplite preset, for example perch_8.",
  )
  return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
  config = runtime.expand_config(runtime.load_yaml(runtime.resolve_project_path(args.config)))
  if args.dataset_root:
    config.setdefault("dataset", {})["root_path"] = args.dataset_root
  if args.perch_repo:
    config.setdefault("perch", {})["repo_root"] = args.perch_repo
  if args.model_choice:
    config.setdefault("perch", {})["model_choice"] = args.model_choice

  root = config.get("dataset", {}).get("root_path")
  resolved_root = resolve_dataset_root(root)
  if resolved_root is not None:
    config.setdefault("dataset", {})["root_path"] = str(resolved_root)
  return config


def add_perch_repo_to_path(config: dict[str, Any]) -> Path | None:
  repo_root = config.get("perch", {}).get("repo_root")
  if not repo_root:
    return None
  path = runtime.resolve_project_path(repo_root)
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))
  return path


def import_status(modules: list[str]) -> dict[str, str]:
  status = {}
  for module in modules:
    try:
      imported = importlib.import_module(module)
      status[module] = f"ok {getattr(imported, '__version__', '')}".strip()
    except Exception as exc:
      status[module] = f"missing: {exc.__class__.__name__}: {exc}"
  return status


def missing_modules(status: dict[str, str]) -> list[str]:
  return [name for name, value in status.items() if not value.startswith("ok")]


def split_labels(dataset_path: Path | None, split: str) -> list[str]:
  if dataset_path is None:
    return []
  split_dir = dataset_path / split
  if not split_dir.is_dir():
    return []
  return sorted(child.name for child in split_dir.iterdir() if child.is_dir())


def dataset_status(config: dict[str, Any]) -> dict[str, Any]:
  dataset_root = config.get("dataset", {}).get("root_path")
  dataset_path = resolve_dataset_root(dataset_root)
  return {
    "root": str(dataset_path) if dataset_path else None,
    "exists": dataset_path.is_dir() if dataset_path else False,
    "train_labels": split_labels(dataset_path, "Train"),
    "validation_labels": split_labels(dataset_path, "Validation"),
  }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
  perch_repo = add_perch_repo_to_path(config)
  core_status = import_status(REQUIRED_MODULES)
  perch_status = import_status(PERCH_MODULES)
  hoplite_status = import_status(HOPLITE_MODULES)
  combined_status = {**core_status, **perch_status, **hoplite_status}
  return {
    "python_supported_by_upstream_pyproject": sys.version_info < (3, 12),
    "python_version": sys.version,
    "perch_repo": str(perch_repo) if perch_repo else None,
    "perch_repo_exists": perch_repo.is_dir() if perch_repo else False,
    "dataset": dataset_status(config),
    "module_status": combined_status,
    "missing_modules": missing_modules(combined_status),
    "imports_ok": not missing_modules(combined_status),
    "model_choice": config.get("perch", {}).get("model_choice"),
  }


def label_map(config: dict[str, Any]) -> dict[str, Any]:
  dataset = dataset_status(config)
  mapping = config.get("label_mapping", {})
  train_labels = dataset["train_labels"]
  validation_labels = dataset["validation_labels"]
  dataset_labels = sorted(set(train_labels) | set(validation_labels))
  missing_mapping = [label for label in dataset_labels if label not in mapping]
  unused_mapping = [label for label in sorted(mapping) if label not in dataset_labels]
  mapped_species = {
    label: {
      "scientific_name": mapping[label].get("scientific_name"),
      "common_name": mapping[label].get("common_name"),
      "perch_strategy": mapping[label].get("perch_strategy", "species_logit"),
    }
    for label in dataset_labels
    if label in mapping
  }
  return {
    "dataset": dataset,
    "num_dataset_labels": len(dataset_labels),
    "dataset_labels": dataset_labels,
    "mapped_species": mapped_species,
    "missing_mapping": missing_mapping,
    "unused_mapping": unused_mapping,
    "mapping_ok": not missing_mapping,
  }


def hoplite_smoke(config: dict[str, Any]) -> dict[str, Any]:
  add_perch_repo_to_path(config)
  status = import_status(HOPLITE_MODULES)
  missing = missing_modules(status)
  if missing:
    return {
      "status": "blocked",
      "module_status": status,
      "missing_modules": missing,
    }

  from perch_hoplite.zoo import model_configs

  model_choice = config.get("perch", {}).get("model_choice", "perch_8")
  try:
    model_config = model_configs.get_preset_model_config(model_choice)
  except Exception as exc:
    return {
      "status": "failed",
      "model_choice": model_choice,
      "error": f"{exc.__class__.__name__}: {exc}",
    }

  public_attrs = {
    name: repr(value)
    for name, value in vars(model_config).items()
    if not name.startswith("_") and isinstance(value, (str, int, float, bool, tuple, list, dict, type(None)))
  }
  return {
    "status": "ok",
    "model_choice": model_choice,
    "model_config_type": type(model_config).__name__,
    "model_config_attrs": public_attrs,
  }


def write_run(config: dict[str, Any], mode: str, results: dict[str, Any]) -> Path:
  run_dir = runtime.make_run_dir(config, mode)
  record = runtime.run_metadata(config, mode)
  record["results"] = results
  runtime.dump_yaml(record, run_dir / "run.yaml")

  module_status = results.get("module_status", {})
  missing = results.get("missing_modules") or missing_modules(module_status)
  dataset = results.get("dataset", {})
  row = {
    "timestamp": record["timestamp"],
    "experiment_id": config["experiment"]["id"],
    "mode": mode,
    "branch": record["git"]["branch"],
    "commit": record["git"]["commit"],
    "run_dir": str(run_dir),
    "perch.imports_ok": results.get("imports_ok"),
    "perch.dataset_exists": dataset.get("exists"),
    "perch.model_choice": results.get("model_choice") or config.get("perch", {}).get("model_choice"),
    "perch.mapped_labels": len(results.get("mapped_species", {})) or None,
    "perch.mapping_ok": results.get("mapping_ok"),
    "perch.missing_modules": ", ".join(missing),
    "perch.status": results.get("status"),
  }
  runtime.append_summary(row, config)
  return run_dir


def main() -> None:
  args = parse_args()
  config = load_config(args)
  if args.mode == "preflight":
    results = preflight(config)
  elif args.mode == "label-map":
    results = label_map(config)
  else:
    results = hoplite_smoke(config)
  run_dir = write_run(config, args.mode, results)
  print(f"Wrote run record: {run_dir / 'run.yaml'}")


if __name__ == "__main__":
  main()
